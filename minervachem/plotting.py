from functools import partial
from io import BytesIO
import math

import cairosvg
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from PIL import Image
import sigfig
from sklearn.metrics import (r2_score, 
                             mean_absolute_error as MAE, 
                             mean_squared_error as MSE)
import warnings

from minervachem.utils.misc import mol_with_atom_index
from minervachem.fingerprinters import GraphletFingerprinter, MorganFingerprinter, RDKitFingerprinter

RMSE = partial(MSE, squared=False)

def remove_bg(im): 
    arr = np.array(im)
    white = (arr.sum(2) == 765)
    alpha = np.zeros_like(white)
    alpha[~white] = 255
    alpha = Image.fromarray(alpha)
    im.putalpha(alpha)
    return


def add_identity(axes, **line_kwargs):
    """https://stackoverflow.com/questions/22104256/"""
    identity, = axes.plot([], [], **line_kwargs)

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes

def add_best_fit(x, y, ax, **kwargs): 
    """Add a line of best fit to a scatterplot of x and y on ax"""
    
    # determine best fit line
    poly = np.polyfit(x, y, 1, full=True)

    slope = poly[0][0]
    intercept = poly[0][1]
    xl = [min(x), max(x)]
    yl = [slope*xx + intercept  for xx in xl]
    ax.plot(xl, yl, **kwargs)
    return ax


def scatter_hist(x, 
                 y, 
                 bins=50, 
                 xlab=None, 
                 ylab=None, 
                 title=None, 
                 ax=None, 
                 fontsize=None,
                 identity=False,
                 best_fit=True,
                 identity_kws=None,
                 best_fit_kws=None,
                 **kwargs):
    """Create a scatter plot (with identity) and a corresponding 2d histogram
    
    """
    if ax is None: 
        fig, ax = plt.subplots(**kwargs)
    ax.set_xlabel(xlab, fontsize=fontsize)
    ax.set_ylabel(ylab, fontsize=fontsize)
    h = ax.hist2d(x, y, bins=bins, norm=matplotlib.colors.LogNorm(), cmap=matplotlib.colormaps['viridis'])
    plt.colorbar(h[3], ax=ax)
    if identity:
        if identity_kws is None: 
            identity_kws = dict(color='black', linestyle='-', label='Identity')
        add_identity(ax, **identity_kws)
    if best_fit: 
        if best_fit_kws is None:
            best_fit_kws = dict(color='black', linestyle='--', label='Best fit')
        add_best_fit(x, y, ax, **best_fit_kws)
    if title:
        ax.set_title(title)
    return ax

def add_textbox(d, 
                ax, 
                x=.05, 
                y=.95, 
                sigfigs=4, 
                notation='standard',
                fontsize=14,
                text_alpha=0.9
               ):
    bbox_params = dict(boxstyle='round',
                       facecolor='white',
                       alpha=text_alpha)
    textparams = dict(transform=ax.transAxes,
                      fontsize=fontsize,
                      verticalalignment='top',
                      bbox=bbox_params)
    text = []
    for k, v in d.items(): 
        v = float(v)
        if math.isnan(v): 
            v = 'NaN'
        else: 
            v = sigfig.round(v, sigfigs=sigfigs, notation=notation)
        
        text.append(f'{k}={v}')
        
    ax.text(x, y,
            '\n'.join(text),
            **textparams)

def parity_plot(true, 
              pred, 
              ax=None, 
              sigfigs=4, 
              notation='standard',
              metrics=None,
              text_loc=(0.05, 0.95),
              legend_loc='lower right',
              text_fontsize=14,
              text_alpha=0.9,
              reverse_met_args=False,
              legend=True,
              identity_kws=None,
              **kwargs):
    """Plot true against pred, with metrics, parity, and a line of best fit
    
    identity_kws get passed to add_identity
    kwargs are passed to scatter_hist
    """
    
    allowed_metrics = {'mae': {'display_name': 'MAE', 
                               'fn': MAE}, 
                       'r2': {'display_name': '$R^2$', 
                              'fn': r2_score}, 
                       'rmse': {'display_name': 'RMSE', 
                                'fn': RMSE}, 
                       'pearsonr': {'display_name': 'Pearson $r$', 
                                    'fn': lambda x, y: pearsonr(x, y)[0]},
                       'spearmanr': {'display_name': r'Spearman $\rho$', 
                                     'fn': lambda x, y: spearmanr(x, y)[0]}
                      }
    
    if metrics == None:
        metrics = 'mae', 'r2', 'rmse'
        
        
    metric_values = {}
    for metric in metrics: 
        if metric in allowed_metrics.keys():
            met_args = (true, pred) if not reverse_met_args else (pred, true)
            metric_values[allowed_metrics[metric]['display_name']] = allowed_metrics[metric]['fn'](*met_args)
        else: 
            warnings.warn(f'"{metric}" not in allowed metrics ({",".join(list(allowed_metrics.keys()))})')
        
    ax = scatter_hist(true, pred, ax=ax, identity_kws=identity_kws, **kwargs)
    if metric_values: 
        x, y = text_loc
        add_textbox(metric_values, ax, x=x, y=y, sigfigs=sigfigs, notation=notation, fontsize=text_fontsize, text_alpha=text_alpha)
        
    if legend: 
        ax.legend(loc=legend_loc)
        
    return ax

def parity_plot_train_test(X, y, model, 
                         title=None, 
                         figsize=(8,3),
                         dpi=None,
                         **kwargs):
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    for i, (subset, x, y) in enumerate(zip(['train', 'test'], X, y)):
        pred = model.predict(x)
        parity_plot(y, pred, ax=axes[i], title=f'{subset.title()} (n={y.shape[0]:,d})', **kwargs)
        if title:
            plt.suptitle(title)
    return axes

def draw_ss(mol, atom_ix, draw_atom_ix=False, ix_note=True):
    """Draw a substructure of a molecule given an iterable of atom indices"""
    #atoms = set([mol.GetAtomWithIdx(i) for i in atom_ix])
    rwmol = Chem.RWMol(mol)
    dos = Chem.Draw.MolDrawOptions()
    if draw_atom_ix and ix_note:
        dos.annotationFontScale=0.7
    if draw_atom_ix:
        rwmol = mol_with_atom_index(rwmol, index_start=1, note=ix_note)
    for atom in reversed(list(rwmol.GetAtoms())):
        if atom.GetIdx() not in atom_ix:
            rwmol.RemoveAtom(atom.GetIdx())
    im = Chem.Draw.MolToImage(rwmol, options=dos)
    return im

def draw_substructure(mol, atom_ix, draw_atom_ix=False, ix_note=True,
                      size=(300, 250), flatten_bonds=True, use_coordgen=True):
    
    """
    Render a molecular substructure defined by a set of atom indices as an SVG.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Input molecule to draw.
    atom_ix : Iterable[int]
        Atom indices (with respect to `mol`) to keep in the rendered substructure.
        Duplicates are ignored.
    draw_atom_ix : bool, default=False
        If True, annotate atoms with their indices in the substructure (after
        deletion/renumbering).
    ix_note : bool, default=True
        If True and `draw_atom_ix` is True, reduce annotation font scale to make
        indices less visually dominant.
    size : tuple[int, int], default=(300, 250)
        Output image size as (width, height) in pixels.
    flatten_bonds : bool, default=True
        If True, remove stereochemistry from the fragment prior to drawing.
    use_coordgen : bool, default=True
        If True, prefer CoordGen for 2D coordinate generation (via
        `rdDepictor.SetPreferCoordGen`).

    Returns
    -------
    str
        SVG text of the rendered substructure.
    """

    keep = sorted(set(atom_ix))
    if not keep:
        raise ValueError("No atoms selected for the substructure.")

    rwm = Chem.RWMol(mol)
    for a in sorted((i for i in range(rwm.GetNumAtoms()) if i not in keep), reverse=True):
        rwm.RemoveAtom(a)
    sub = rwm.GetMol()
    if sub.GetNumAtoms() == 0:
        raise ValueError("Fragment is empty after deletion.")
    
    if flatten_bonds:
        Chem.RemoveStereochemistry(sub)
    if sub.GetNumAtoms() > 1:
        sub.RemoveAllConformers()
        rdDepictor.SetPreferCoordGen(bool(use_coordgen))
        rdDepictor.Compute2DCoords(sub)

    d2d = rdMolDraw2D.MolDraw2DSVG(*size)
    opts = d2d.drawOptions()
    if draw_atom_ix:
        opts.addAtomIndices = True
        if ix_note:
            opts.annotationFontScale = 0.7
            
    rdMolDraw2D.PrepareAndDrawMolecule(d2d, sub)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()

def pil_from_svg(svg_text: str, *, scale=1.0, output_width=None, output_height=None):
    """
    Convert an SVG string into a Pillow (PIL) RGBA image.

    Parameters
    ----------
    svg_text : str
        SVG markup as a Unicode string.
    scale : float, default=1.0
        Uniform scaling factor passed to CairoSVG during rasterization.
    output_width : int | None, default=None
        Target output width in pixels. If provided, overrides the SVG's intrinsic width.
    output_height : int | None, default=None
        Target output height in pixels. If provided, overrides the SVG's intrinsic height.

    Returns
    -------
    PIL.Image.Image
        The rasterized image in RGBA mode.
    """
    png_bytes = cairosvg.svg2png(bytestring=svg_text.encode("utf-8"),
                                 scale=scale,
                                 output_width=output_width,
                                 output_height=output_height)
    return Image.open(BytesIO(png_bytes)).convert("RGBA")

def plot_fingerprint(mol,
                     fingerprinter,
                     ncol=3,
                     figsize=None,
                     decreasing=True,
                     show_count=True,
                     show_bit_ids=True,
                     show_size=True,
                     only_found=True):
    """Plot all of the substructures induced in mol by fingerprinter.

    Note that if the fingerprinter produces folded bits, this will only show the first one found.


    Args:
        mol: (rdkit.Molecule) the molecule to be fingerprinted
        fingerprinter: a minerva.fingerprinters.Fingerprinter object
        ncol: (int) the number of columns in the plot (n_rows = ceil(n_fragments / n_cols))
        figsize: passed to plt.subplots
        decreasing: (bool) sort fragments in order of decreasing size (if False, sort in increasing order)
        show_count: (bool) show the count of each fragment in the molecule
        show_bit_ids: (bool) show the ID of each bit
        show_size: (bool) show the size of the fragment (this is also included in the bit ID)
    """

    fp, bi = fingerprinter(mol)
    if only_found:
        fp = {k: v for k, v in fp.items() if v != 0}
        bi = {k: v for k, v in bi.items() if v != []}
    n_bits = len(bi.keys())

    if n_bits == 0:
        print('Found no fingerprint bits, exiting')
        return
    nrow = int(np.ceil(len(bi.keys()) / ncol))
    if figsize is None:
        figsize = (10, 5*nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)


    # helper plotting functions for each type of fingerprinter
    _bi = {k[1]: v for k, v in bi.items()}
    if isinstance(fingerprinter, GraphletFingerprinter):
        def draw_fn(mol, bit):
            return draw_ss(mol, bi[bit][0])
        size_name = 'n'
        fingerprinter_name = 'Graphlet Fingerprint'
    elif isinstance(fingerprinter, MorganFingerprinter):
        def draw_fn(mol, bit):
            return Chem.Draw.DrawMorganBit(mol, bit[1], _bi)
        size_name = 'r'
        fingerprinter_name = 'Morgan Fingerprint'
    elif isinstance(fingerprinter, RDKitFingerprinter):
        def draw_fn(mol, bit):
            return Chem.Draw.DrawRDKitBit(mol, bit[1], _bi)
        size_name = 'l'
        fingerprinter_name = 'RDKit Fingerprint'
    else:
        raise ValueError(f'Got unsupported fingerprinter of type ({type(fingerprinter)})')

    axes = axes.ravel()
    bit_ids_by_l = sorted(bi.keys(), key=lambda k: k[0], reverse=decreasing)
    for i, bit in enumerate(bit_ids_by_l):
        plt.sca(axes[i])
        im = draw_fn(mol, bit)
        plt.imshow(im)
        plt.axis('off')

        plt_text = []
        if show_bit_ids:
            plt_text.append(f'Bit ID: {bit}')
        if show_size:
            plt_text.append(f'{size_name} = {bit[0]}')
        if show_count:
            plt_text.append(f'count = {fp[bit]}')
        plt.text(0, 10, '\n'.join(plt_text))

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    #plt.suptitle(f'{fingerprinter_name} substructures for {Chem.MolToSmiles(mol)}')
    plt.tight_layout()
    return fig, axes
