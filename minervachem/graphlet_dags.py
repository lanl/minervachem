"""Handles molecular substructure relationships

"""

from collections import defaultdict
import numpy as np
import networkx as nx
from matplotlib import cm, colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy

import cairosvg
from PIL import Image, ImageChops
from io import BytesIO
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

from xml.etree import ElementTree as ET

from rdkit import Chem
from rdkit.Chem import MolToSmiles
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import MolToSmiles

from minervachem.plotting import draw_ss, remove_bg

def group_atoms_by_size(bi, decreasing=True):
    """Partitions a nested iterable of atomindices by the length of the inner iterables"""
    out = defaultdict(lambda: [])
    for bit_id, atoms in bi.items(): 
        size, hash_ = bit_id
        atoms = map(lambda a: tuple(sorted(a)), atoms)
        out[size].extend(atoms)
        
    sizes = list(sorted((out.keys())))
    if decreasing: 
        sizes = list(reversed(sizes))
    return sizes, dict(out)


class GraphletDAG:

    def __init__(self, 
                 mol, 
                 fingerprinter=None,
                 bi=None, 
                 layerwise=True, 
                 bit_ids=None,
                 coef=None,
                 bit_to_coef_map=None):
        """The Directed Acyclic Graph (DAG) of inclusion relationships between molecular graphlets

        A utility for visualizing this graph and performing prediction projections
        onto atom and bond level fragments.

        You must construct this either with a fingerprinter or a bit_info (bi) dictionary from a fingerprinter. 

        mol: rdkit molecule
        bi: bit info -- a dict mapping fingerprint bit ids to iterables of atom indices. Overrides fingerprinter
        fingerprinter: a fingerprinter objec.t overrided by bi
        layerwise: bool, whether to perform projection layerwise or directly
        bit_ids: the labels for coef. must be given with coef
        coef:    an np.array of coefficients, some of which will be assigned to DAG nodes. must be given with bit_ids
        bit_to_coef_map: mapping from bit IDs to coefficients. Overrides bit_ids and coef arguments
        """

        self.mol = mol
        self.bi = bi
        self.layerwise = layerwise
        self.fingerprinter = fingerprinter
        self.bit_ids = bit_ids
        self.coef = coef
        
        # compute the bi if we can and were not given one
        if self.fingerprinter is not None and self.bi is None: 
            _, self.bi = fingerprinter(mol)
        
        # handle bit to coef map. If one is provided, it overrides computing one
        if (bit_to_coef_map is None 
            and bit_ids is not None 
            and coef is not None): 
            bit_to_coef_map = dict(zip(bit_ids, coef))
        self.bit_to_coef_map = bit_to_coef_map

        # these define the layers of the DAG
        self.sizes, self.atoms_by_size = group_atoms_by_size(self.bi)

        # Construct the graph. This will store a networkx digraph object in self.G
        self.G = None
        self._build_ssg()

        # we associate the coefs with the appropriate nodes in self.G if we have them
        if self.bit_to_coef_map is not None:
            self._add_coefs_to_nodes()
        
        # This will be set during a projection 
        self.projected_coefs = None

        # keep track of projection since we do not allow iterating projection
        self._has_been_projected = False
        

    def _create_graph_nodes(self):
        """Creates a node in a nx graph for every atom-induced subgraph"""

        G = nx.DiGraph()

        for bit, atom_sets in self.bi.items():

            size = bit[0]
            is_atom = size == 1
            is_bond = size == 2

            for atoms in atom_sets:
                atoms = tuple(sorted(atoms))  # identify nodes by the atoms
                # create the node
                G.add_node(atoms, bit=bit, size=size)
                # add additional properties for atoms and bonds
                if is_atom:
                    # add the valence of the atom
                    atom = atoms[0]
                    valence = sum(bond.GetBondTypeAsDouble() for bond in self.mol.GetAtoms()[atom].GetBonds())
                    G.add_node(atoms, valence=valence, atom_id=atom)
                if is_bond:
                    # add the bond ID and the bond order
                    bond = self.mol.GetBondBetweenAtoms(*atoms)
                    bond_id = bond.GetIdx()
                    bond_order = bond.GetBondTypeAsDouble()
                    G.add_node(atoms, bond_id=bond_id, bond_order=bond_order)
        self.G = G
        return

    def _create_graph_edges(self):
        """create the substructure inclusion edges. if self.layerwie, only between size n and n-1"""

        for i, size in enumerate(self.sizes[:-1]):
            for atoms in self.atoms_by_size[size]:
                # either just the next smallest size or all smaller sizes
                next_sizes = [self.sizes[i + 1]] if self.layerwise else self.sizes[i + 1:]
                for next_size in next_sizes:
                    for atoms_prime in self.atoms_by_size[next_size]:
                        if set(atoms_prime).issubset(set(atoms)):
                            self.G.add_edge(atoms, atoms_prime)

    def _build_ssg(self):
        self._create_graph_nodes()
        self._create_graph_edges()
        return

    def __repr__(self):
        return f'GraphletDAG({MolToSmiles(self.mol)}, layerwise={self.layerwise})'

    def draw(self,
             figsize=(12, 8),
             dpi=300,
             align='horizontal',
             label_nodes=True,
             draw_nodes_kws=None,
             draw_edges_kws=None,
             draw_fragments=False,
             draw_atom_ix=True,
             ):
        """Draw the DAG of inclusion relationships between molecular graphlets
        
        figsize: tuple -- matplotlib figize
        dpi: matplotlib dpi
        align: vertical or horizontal 
        label_nodes: bool, whether to include node labels on the plot (labels are subsets of atom indices, which are truncated)
        draw_nodes_kws: dict, passed to networkx.draw_networkx_nodes
        draw_edges_kws: dict, passed to networkx.draw_networkx_edges
        draw_fragments: bool, whether or not to draw the molecular graphlets on the DAG nodes
        draw_atom_ix: bool, if draw_fragments, whether to show the atom indices
        """

        if draw_edges_kws is None:
            draw_edges_kws = dict(width=5, arrowsize=30, node_size=3000, edge_color='k')
        if draw_nodes_kws is None:
            draw_nodes_kws = dict(alpha=.75, node_size=2000, node_color='#1f78b4')

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        pos = multipartite_layout(self.G, 'size', align=align)
        nx.draw_networkx_edges(self.G, pos, **draw_edges_kws)

        if draw_fragments: 
            # transform from display to figure coordinates 
            tr_figure = ax.transData.transform
            tr_axes = fig.transFigure.inverted().transform

            # Select the size of the image (relative to the X axis)
            icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.1
            icon_center = icon_size / 2.0
            # Add the respective image to each node
            for n in self.G.nodes:
                xf, yf = tr_figure(pos[n])
                xa, ya = tr_axes((xf, yf))
                a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
                im = draw_ss(self.mol, n, draw_atom_ix=draw_atom_ix)
                remove_bg(im)
                a.imshow(im)
                plt.axis('off')

        else:
            nx.draw_networkx_nodes(self.G, pos, **draw_nodes_kws)
            if label_nodes:
                nx.draw_networkx_labels(self.G,
                                        pos,
                                        self.node_labels,
                                        bbox=dict(color='w', alpha=.75)
                                        )
        return fig, ax

    @property
    def node_labels(self):
        labels = {}
        for node in self.G:
            if len(node) > 3:
                label = '{' + str(node[0]) + ',...,' + str(node[-1]) + '}'
            else:
                label = str(set(node))
            labels[node] = label
        return labels

    def _add_coefs_to_nodes(self):
        for node in self.G.nodes:
            coef = self.bit_to_coef_map.get(self.G.nodes[node]['bit'], 0)
            self.G.add_node(node, coef=coef)
        return

    def _project_from_layer(self,
                            l,
                            direction='down',
                            final_target_layer=None,
                            weight_by_bond_order=True
                            ):
        """Project attrs from layer l to l +/- 1 in layerwise case to final_target_layer otherwise"""
        G = self.G
        layer_l = self.atoms_by_size[l]
        layer_next = G.succ if direction == 'down' else G.pred
        if not self.layerwise:
            # then only project to the target layer
            _layer_next = {}
            for k, v in layer_next.items():
                if len(k) == l:
                    _layer_next[k] = {x: y for x, y in layer_next[k].items() if len(x) == final_target_layer}

            layer_next = _layer_next

        if direction == 'up' and l > 1:
            raise RuntimeError('only projecting up to bonds is supported')

        # for each node in this layer
        for node_l in layer_l:
            # grab the coef and look at the children
            coef_l = G.nodes[node_l]['coef']
            children = dict(layer_next[node_l])

            # for each child
            for child in children.keys():
                if direction == 'up' and weight_by_bond_order:
                    # up projection considers bond order and valence
                    # currently only implemented for layer 1 to 2 projection
                    bond_order = G.nodes[child]['bond_order']
                    atom_valence = G.nodes[node_l]['valence']
                    delta = coef_l * (bond_order / atom_valence)
                    # also reverse the edge direction to show direction of projection
                    G.remove_edge(child, node_l)
                    G.add_edge(node_l, child)
                else:
                    # always evenly divide among children on down projection
                    delta = coef_l / len(children)
                coef_new = G.nodes[child]['coef'] + delta
                G.add_node(child, coef=coef_new)

    def _undo_projection(self):
        self.G = self._G
        self._has_been_projected = False

    def project_to_layer(self, layer, weight_by_bond_order=True):
        """Perform the projection of coefficients on nodes to either atoms or bonds. 

        Note that the original state of the DAG is stored, so you can perform this more than once with no ill-effects.

        layer: which layer to project to (currently must be 1 or 2)
        weight_by_bond_order: whether to use the bond order / total_bond_order to weight the up projections
        """

        if self.bit_to_coef_map is None: 
            raise RuntimeError(("GraphletDAG has no node coefficients, cannot perform projection. "
                                "Please pass bit_to_coef_map to __init__ method if you wish to project coefficients."))
        # restore and backup original state
        if self._has_been_projected:
            self._undo_projection()
        self._G = deepcopy(self.G)
        self._has_been_projected = True

        if layer > 2:
            raise ValueError('Currently only support projection to layers 1 or 2')

        # do all of the down projections
        for l in self.sizes[:-layer]:
            self._project_from_layer(l, final_target_layer=layer)

        # and all of the up projections
        if layer == 2:
            self._project_from_layer(1, 'up', final_target_layer=layer, weight_by_bond_order=weight_by_bond_order)

        # gather projected coefficients into an array
        layer_nodes = [n for n in self.G.nodes if self.G.nodes[n]['size'] == layer]
        for node in layer_nodes:
            id_key = 'atom_id' if layer == 1 else 'bond_id'
            layer_nodes = sorted(layer_nodes, key=lambda node: self.G.nodes[node][id_key])
            coefs = [self.G.nodes[node]['coef'] for node in layer_nodes]

        self.projected_coefs = np.asarray(coefs)
        return self.projected_coefs.copy()


def group_bi_by_size(bi, reverse=True):
    out = defaultdict(lambda: [])
    for k, v in bi.items(): 
        size, hash_ = k
        out[size].append(k)
        
    sizes = list(sorted((out.keys())))
    if reverse: 
        sizes = list(reversed(sizes))
    return sizes, dict(out)
            
def get_substruct_scalar_mappable(attrs, cmap=None, normalizer=None): 
    attrs = np.asarray(attrs)
    cmap = plt.get_cmap(cmap)
    if normalizer is None:
        normalizer = colors.Normalize()
    elif isinstance(normalizer, str): 
        if normalizer == 'centered':
            normalizer = colors.CenteredNorm(vcenter=0)
        else: 
            raise ValueError(f"Unknown string for normalizer ({normalizer})")
    elif not isinstance(normalizer, colors.Normalize): 
        raise ValueError((f"Unsupported type ({type(normalizer)} for normalizer. "
                          "Must be string or matplotlib.colors.Normalize"))

    #attrs = normalizer(attrs)
    sm = cm.ScalarMappable(
          normalizer,
          cmap
    )
    # matplotlib seems to memorize what we give it for the first time
    _, sm = get_substruct_colors(attrs, sm)
    return sm


def get_substruct_colors(ss_attrs, 
                         sm=None, 
                         cmap=None,
                         normalizer=None
                       ): 
    if sm is None: 
        sm = get_substruct_scalar_mappable(ss_attrs, cmap, normalizer=normalizer)
        
    colors_rgb = sm.cmap(sm.norm(ss_attrs))[:, :-1]
    colors_rgb = [tuple(row) for row in colors_rgb]
    return colors_rgb, sm

class CSSStyle:
    
    def __init__(self, data):
        self._dict = {}
        self._dict.update(self._parse_string(data))
        
    def __getattr__(self, attrname):
        return getattr(self._dict, attrname)
    
    def _parse_string(self, style):
        return dict(map(lambda x: x.split(':'), style.split(';')))
    
    def tostring(self):
        return ";".join(map(lambda x: "{0}:{1}".format(*x), self._dict.items()))
    
    def update(self, k, v): 
        self.dict[k] = v
        
def set_stroke_to_black(style): 
    style = CSSStyle(style)
    for k in [
                'stroke', 
                #'fill',
             ]: 
        if k in style._dict.keys(): 
            style._dict[k] = '#000000'
    return style.tostring()

def set_fonts_to_black(mol_css_str): 
    et = ET.fromstring(mol_css_str)
    paths = et.findall('{http://www.w3.org/2000/svg}path')
    for path in paths: 
        style = path.get('style')
        if style:
            new_style = set_stroke_to_black(style)
            path.set('style', new_style)
        if path.get('fill'): 
            path.set('fill', '#000000')
    return ET.tostring(et, encoding='unicode')

def set_svg_fonts_to_black(svg_text): 
    mol_count = 0 
    et = ET.fromstring(svg_text)
    for path in et.findall('{http://www.w3.org/2000/svg}path'):
        _class = path.get('class')
        if _class == None: 
            mol_count += 1

        if mol_count > 0: 
            style = path.get('style')
#             if style:
#                 new_style = set_stroke_to_black(style)
#                 path.set('style', new_style)
            if path.get('fill'): 
                path.set('fill', '#000000')
    svg_text = ET.tostring(et, encoding='unicode')
    return svg_text

def get_draw_args(mol, mol_attrs, sm, level): 
    if mol_attrs is not None:
        ss_colors, sm_i = get_substruct_colors(mol_attrs, sm)
        ss_colors = {i: [c] for i, c in enumerate(ss_colors)}
        arads = {i: 0.3 for i in range(mol.GetNumAtoms())}
        # todo mt.2022.06.22: encapsulate this block? 
        if level == 1: # atom 
            draw_args = mol, '', ss_colors, {}, arads, {}
        else: 
            draw_args = mol, '', {i: [(0.8,0.8,0.8)] for i in range(mol.GetNumAtoms())}, ss_colors, arads, {}
    else: 
        draw_args = mol, '', {}, {}, {}, {}
    return draw_args



def draw_mol_with_colors(mol, 
                         mol_attrs, 
                         sm, 
                         level, 
                         size, 
                         dpi, 
                         font_black=False): 
    """returns an image"""
    draw_args = get_draw_args(mol, mol_attrs, sm, level)
    d2d = rdMolDraw2D.MolDraw2DSVG(*size)
    d2d.DrawMoleculeWithHighlights(*draw_args)
    d2d.FinishDrawing()
    drawing_str = d2d.GetDrawingText()
    if font_black: 
        drawing_str = set_fonts_to_black(drawing_str)
    img = cairosvg.svg2png(drawing_str, 
                           dpi=dpi)
    img = Image.open(BytesIO(img))

    # add an alpha layer 
    # note: this is very conservative and will leave some pixels that may have been best alphaed out
    # it seems that the background color option in the svg2png method wont help us. 
    arr = np.array(img)
    white = (arr.sum(2) == 765)
    alpha = np.zeros_like(white)
    alpha[~white] = 255
    alpha = Image.fromarray(alpha)
    img.putalpha(alpha)
    return img


def draw_projected_coefs(graphlet_dags,
                         level=1,
                         share_cbar=True, 
                         cbar_label=None,
                         figsize=None, 
                         svgsize=(500, 500), 
                         ncol=3,
                         titles=None, 
                         annotate=True,
                         dpi=300,
                         cmap='summer',
                         all_attrs=None,
                         cbar_orient='v',
                         prepend_mol=False,
                         font_black=False,
                         normalizer=None,
                        ): 
    """Visualize atom- or bond- level projection values of GraphletDAG objects

    Plots on a matplotlib figure.

    graphlet_dags: GraphletDAG or list of substucturegraph
    level: 1 or 2 -- which level to project the coefficients to

    """
    
    
    # if we get one ssg, pack it in a list so we can iterate over it
    if isinstance(graphlet_dags, GraphletDAG): 
        graphlet_dags = [graphlet_dags]
    graphlet_dags = deepcopy(graphlet_dags)
    
    if prepend_mol: 
        graphlet_dags.insert(0, graphlet_dags[0])
    
    # wrap mols into columns
    nmol = len(graphlet_dags)
    if nmol < ncol: 
        ncol = nmol
    nrow = int(np.ceil(nmol / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    
    # make sure axes is always a 1D iterable
    if nmol == 1: 
        axes = [axes]
    else: 
        axes = axes.ravel()
    
    # if we want to share the colors, build one scalar mappable for all of the mols
    # else the cmap will be constructed downstream per molecule
    sm = None
    if share_cbar: 
        all_attrs = np.hstack([mol.project_to_layer(level) for mol in graphlet_dags])
        sm = get_substruct_scalar_mappable(all_attrs, cmap, normalizer)
        
    for i, ssg in enumerate(graphlet_dags):
        mol_attrs = ssg.project_to_layer(level)
        if i == 0 and prepend_mol: 
            mol_attrs = None
        if not share_cbar: 
            sm = get_substruct_scalar_mappable(mol_attrs, cmap, normalizer)
        
        mol = Chem.Mol(ssg.mol)
        plt.sca(axes[i])
        if annotate: 
            add_annotations(mol, mol_attrs, level)
        img = draw_mol_with_colors(mol, 
                                    mol_attrs, 
                                    sm, 
                                    level, 
                                    svgsize, 
                                    dpi, 
                                    font_black=font_black)
        #img = trim(img)
        plt.imshow(img)
        plt.axis('off')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        if titles is not None:
            plt.title(titles[i])
        
        
        # Draw colorbars if unshared or at the end of a row
        end_of_row = (i+1) % ncol == 0
        draw_cbar = (not share_cbar) or end_of_row
        if draw_cbar: 
            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax)
            if cbar_orient == "v": 
                cax_location = "right"
                cbar_orientation = "vertical"
                cbar_label_rotation = 270
            else: 
                cax_location = "bottom"
                cbar_orientation = "horizontal"
                cbar_label_rotation = 0
            cax = divider.append_axes(cax_location, size="5%", pad=0.05)
            cbar = plt.colorbar(sm, cax=cax, orientation=cbar_orientation)
            if cbar_label is not None:
                cbar.ax.set_ylabel(cbar_label, 
                                    rotation=cbar_label_rotation, 
                                    labelpad=15)
            cax.tick_params(labelsize=16)
    for j in range(i+1, len(axes)): 
        axes[j].set_visible(False)
        
    return fig, axes


def draw_projected_coefs_svg(graphlet_dags,
                             level=1,
                             ncol=3,
                             prepend_mol=False,
                             fonts_black=True,
                             svg_size=(400, 100), # individual
                             cmap='summer_r',
                             legends=None,
                             hilight_radii=0.4,
                             normalizer=None
                             ): 
    if isinstance(graphlet_dags, GraphletDAG): 
        graphlet_dags = [graphlet_dags]
    graphlet_dags = deepcopy(graphlet_dags)
    
    if prepend_mol: 
        graphlet_dags.insert(0, deepcopy(graphlet_dags[0]))
    
    # get the scalar mappable that will color the atoms/bonds
    all_attrs = [mol.project_to_layer(level) for mol in graphlet_dags]
    sm = get_substruct_scalar_mappable(np.hstack(all_attrs), cmap, normalizer=normalizer)
    
    # get the draw args for all of the mols
    draw_args = defaultdict(lambda: [])
    if legends is not None: 
        draw_args['legends'] = legends
    mols = []
    for i, ssg in enumerate(graphlet_dags): 
        mol = ssg.mol
        mols.append(mol)
        
        ss_colors, _ = get_substruct_colors(all_attrs[i], sm)
        ss_colors = {i: c for i, c in enumerate(ss_colors)}
        if not (prepend_mol and i==0):
            if level == 1: # atom 
                atom_colors, bond_colors = ss_colors, {i: (1,1,1) for i in range(mol.GetNumBonds())}
                #add_annotations(mol, all_attrs[i], level)
            else: 
                dummy_colors = {i: (0.8,0.8,0.8) for i in range(mol.GetNumAtoms())}
                atom_colors, bond_colors = dummy_colors, ss_colors
                
        else: 
            atom_colors = {i: (1,1,1) for i in range(mol.GetNumAtoms())}
            bond_colors = {i: (1,1,1) for i in range(mol.GetNumBonds())}
        
        #draw_args['highlight_radii'].append({i: 0.3 for i in range(mol.GetNumAtoms())})
        draw_args['highlightAtomColors'].append(atom_colors)
        draw_args['highlightBondColors'].append(bond_colors)
        draw_args['highlightAtoms'].append(list(range(mol.GetNumAtoms())))
        draw_args['highlightAtomRadii'].append({i: hilight_radii for i in range(mol.GetNumAtoms())})
    
        
    # wrap mols into columns
    nmol = len(graphlet_dags)
    if nmol < ncol: 
        ncol = nmol
    nrow = int(np.ceil(nmol / ncol))
    
    # create initial drawing object
    w, h = svg_size
    d2d = rdMolDraw2D.MolDraw2DSVG(w*ncol, h*nrow, w, h)
    
    draw_args = {k: tuple(v) for k, v in draw_args.items()}
    d2d.drawOptions().highlightBondWidthMultiplier = 15
    #d2d.drawOptions().scalehighlightBondWidth = True
    #d2d.drawOptions().fillHilights=True
    #d2d.drawOptions().fixedBondLength = True
    #d2d.drawOptions().scaleBondWidth = True
    d2d.DrawMolecules(mols, **draw_args)
    d2d.FinishDrawing()
    svg_text = d2d.GetDrawingText()
    if fonts_black: 
        svg_text = set_svg_fonts_to_black(svg_text)
    
    return svg_text, all_attrs


def add_annotations(mol, mol_attrs, level): 
    
    
    if level == 2: 
        _it, prop = mol.GetBonds, 'bondNote'
    else: 
        _it, prop = mol.GetAtoms, 'atomNote'
    
    for j, component in enumerate(_it()):
        attr = float(mol_attrs[j])
        component.SetProp(prop, str(
#             sigfig.round(attr, 
#                          2, 
#                          notation='scientific')
            round(attr, 2)
        ))

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def save_svg_text(svg_text, file_name):
    """give file name as either .png or .pdf"""
    suffix = file_name[-3:]
    if not suffix in ['png', 'pdf']:
        raise ValueError('can only save pdf or png')
    with open(file_name, 'w+') as f: 
        f.write(svg_text)
    # convert to pdf
    if suffix=='pdf':
        drawing = svg2rlg(file_name)
        renderPDF.drawToFile(drawing, file_name)
    

## below functions reproduced and modified from networkx

def multipartite_layout(G, subset_key="subset", align="vertical", scale=1, center=None):
    """Position nodes in layers of straight lines.
    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.
    subset_key : string (default='subset')
        Key of node data to be used as layer subset.
    align : string (default='vertical')
        The alignment of nodes. Vertical or horizontal.
    scale : number (default: 1)
        Scale factor for positions.
    center : array-like or None
        Coordinate pair around which to center the layout.
    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node.
    Examples
    --------
    >>> G = nx.complete_multipartite_graph(28, 16, 10)
    >>> pos = nx.multipartite_layout(G)
    Notes
    -----
    This algorithm currently only works in two dimensions and does not
    try to minimize edge crossings.
    Network does not need to be a complete multipartite graph. As long as nodes
    have subset_key data, they will be placed in the corresponding layers.
    """
    import numpy as np

    if align not in ("vertical", "horizontal"):
        msg = "align must be either vertical or horizontal."
        raise ValueError(msg)

    G, center = _process_params(G, center=center, dim=2)
    if len(G) == 0:
        return {}

    layers = {}
    for v, data in G.nodes(data=True):
        try:
            layer = data[subset_key]
        except KeyError:
            msg = "all nodes must have subset_key (default='subset') as data"
            raise ValueError(msg)
        layers[layer] = [v] + layers.get(layer, [])

    # Sort by layer, if possible
    try:
        layers = sorted(layers.items())
    except TypeError:
        layers = list(layers.items())

    pos = None
    nodes = []
    width = len(layers)
    for i, (_, layer) in enumerate(layers):
        layer = list(sorted(layer))
        height = len(layer)
        xs = np.repeat(i, height)
        ys = np.arange(0, height, dtype=float)
        offset = ((width - 1) / 2, (height - 1) / 2)
        layer_pos = np.column_stack([xs, ys]) - offset
        if pos is None:
            pos = layer_pos
        else:
            pos = np.concatenate([pos, layer_pos])
        nodes.extend(layer)
    pos = rescale_layout(pos, scale=scale) + center
    if align == "horizontal":
        pos = pos[:, ::-1]  # swap x and y coords
    pos = dict(zip(nodes, pos))
    return pos

def _process_params(G, center, dim):
    # Some boilerplate code.
    import numpy as np

    if not isinstance(G, nx.Graph):
        empty_graph = nx.Graph()
        empty_graph.add_nodes_from(G)
        G = empty_graph

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)

    if len(center) != dim:
        msg = "length of center coordinates must match dimension of layout"
        raise ValueError(msg)

    return G, center

def rescale_layout(pos, scale=1):
    """Returns scaled position array to (-scale, scale) in all axes.
    The function acts on NumPy arrays which hold position information.
    Each position is one row of the array. The dimension of the space
    equals the number of columns. Each coordinate in one column.
    To rescale, the mean (center) is subtracted from each axis separately.
    Then all values are scaled so that the largest magnitude value
    from all axes equals `scale` (thus, the aspect ratio is preserved).
    The resulting NumPy Array is returned (order of rows unchanged).
    Parameters
    ----------
    pos : numpy array
        positions to be scaled. Each row is a position.
    scale : number (default: 1)
        The size of the resulting extent in all directions.
    Returns
    -------
    pos : numpy array
        scaled positions. Each row is a position.
    See Also
    --------
    rescale_layout_dict
    """
    # Find max length over all dimensions
    lim = 0  # max coordinate for all axes
    for i in range(pos.shape[1]):
        pos[:, i] -= pos[:, i].mean()
        lim = max(abs(pos[:, i]).max(), lim)
    # rescale to (-scale, scale) in all directions, preserves aspect
    if lim > 0:
        for i in range(pos.shape[1]):
            pos[:, i] *= scale / lim
    return pos
