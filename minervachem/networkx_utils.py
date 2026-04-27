import matplotlib.pyplot as plt
from architector import io_ptable
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import io
import re
import networkx as nx
import numpy as np

def draw_mol_nxgraph(G, figsize=(6, 5), node_scale=800, font_scale=1.6, edge_width=2, just_show=False):
    """Draw molecule graph with atom symbols and bond type labels, colored by atom type."""

    element_colors = {
        "H": "#FFFFFF",  # White
        "C": "#B3B3B3",  # Grey
        "N": "#6C91F1",  # Blue
        "O": "#FF0D0D",  # Red
        "F": "#90E050",  # Green
        "Cl": "#1FF01F", # Greenish
        "Br": "#A62929", # Dark red/brown
        "I": "#940094",  # Violet
        "P": "#FF8000",  # Orange
        "S": "#FFFF30",  # Yellow
        "B": "#FFB5B5",  # Pink
        "Si": "#F0C8A0", # Beige
        "Fe": "#E06633", # Orange-brown
        "Zn": "#7D80B0", # Light purple
    }

    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)

    # Node labels: element symbols or sybyl types
    node_labels = {n: (G.nodes[n].get("element") or G.nodes[n].get("sybyl_type").split('.')[0]) for n in G.nodes}
    
    node_colors = [
        element_colors.get(G.nodes[n].get("element"), "lightgray") for n in G.nodes
    ]
    edge_labels = {(i, j): G.edges[i, j].get("raw_type", "") for i, j in G.edges}

    nx.draw(
        G, pos,
        with_labels=False,
        node_size=node_scale,
        node_color=node_colors,
        edgecolors="black",
        width=edge_width,
        ax=ax
    )
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_weight="bold", font_size=12 * font_scale, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", font_size=8 * font_scale, ax=ax)

    plt.axis("equal")
    plt.tight_layout()

    if just_show:
        plt.show()
        return
    else:
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        buf.seek(0)
        im = Image.open(buf)
        return im
    

_ELEMENTS = io_ptable.elements
_METALS = io_ptable.all_metals

def _guess_element(atom_name: str, atom_type: str | None) -> str | None:
    if atom_type:
        t0 = atom_type.split('.')[0]
        cand = t0[0].upper() + (t0[1:].lower() if len(t0) > 1 else "")
        if cand in _ELEMENTS:
            return cand
    m = re.match(r"([A-Za-z]{1,2})", atom_name or "")
    if m:
        cand = m.group(1)[0].upper() + (m.group(1)[1:].lower() if len(m.group(1)) == 2 else "")
        if cand in _ELEMENTS:
            return cand
    return None

def _bond_order_and_flags(bond_type: str | None):
    if not bond_type:
        return None, False
    bt = bond_type.lower()
    aromatic = (bt == "ar")
    if bt in ("1", "single"):   order = 1
    elif bt in ("2", "double"): order = 2
    elif bt in ("3", "triple"): order = 3
    elif bt == "ar":            order = 1.5
    elif bt == "am":            order = 1 # amide single
    elif bt == "un":            order = 0.5 # un should mostly be dative
    else:                       order = None
    return order, aromatic

def mol2_to_networkx(mol2_text: str, explicit_h: bool = False, topology_only: bool = False) -> nx.Graph:
    """
    Parse MOL2 (string) -> NetworkX Graph

    Node attrs (explicit_h=True):
        - name, element, sybyl_type, x,y,z, charge, subst_id, subst_name
        - atom_key = (symbol, atom_type)
        - original_index (1-based from MOL2)

    Node attrs (explicit_h=False; hydrogens collapsed):
        - all of the above EXCEPT element/name for removed H nodes
        - h_count (# directly bonded H)
        - atom_key = (symbol, atom_type, h_count)
        - original_index maps to original heavy-atom id list (1-based indices of heavy atoms)

    Edge attrs (both modes):
        - order (float|None), aromatic (bool), raw_type (str), bond_key (==raw_type)
        - has_h (bool)  # True if either endpoint in the original MOL2 bond was a hydrogen
        - original_indices = (i1, j1)  # original 1-based atom ids
    """
    atoms_tmp = {}  # id1 -> dict
    bonds_tmp = []  # (i1, j1, raw_type)

    in_atoms = in_bonds = False
    for raw in io.StringIO(mol2_text):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("@<TRIPOS>") or line.startswith("<TRIPOS>"):
            tag = line.replace("@", "").upper()
            in_atoms = (tag == "<TRIPOS>ATOM")
            in_bonds = (tag == "<TRIPOS>BOND")
            continue
        if in_atoms:
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                atom_id = int(parts[0])
                name = parts[1]
                x, y, z = map(float, parts[2:5])
                sybyl_type = parts[5]
                subst_id = int(parts[6])
                subst_name = parts[7]
                charge = float(parts[8]) if len(parts) >= 9 else 0.0
            except Exception:
                continue
            elem = _guess_element(name, sybyl_type)
            atoms_tmp[atom_id] = dict(
                name=name, element=elem, sybyl_type=sybyl_type,
                x=x, y=y, z=z, charge=charge,
                subst_id=subst_id, subst_name=subst_name,
                original_index=atom_id
            )
        elif in_bonds:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                i1 = int(parts[1]); j1 = int(parts[2]); btype = parts[3]
                if topology_only:
                    btype = '1'
            except Exception:
                continue
            bonds_tmp.append((i1, j1, btype))

    # classify hydrogens
    is_h = {aid: (atoms_tmp[aid].get("element") == "H") for aid in atoms_tmp}
    heavy_ids = [aid for aid in sorted(atoms_tmp) if not is_h[aid]]
    heavy_reindex = {aid: idx for idx, aid in enumerate(heavy_ids, start=1)}  # 1..N

    # precompute per-atom H counts (in original indexing)
    h_count_map = {aid: 0 for aid in atoms_tmp}
    for i1, j1, _ in bonds_tmp:
        if is_h.get(i1, False) and not is_h.get(j1, False):
            h_count_map[j1] += 1
        elif is_h.get(j1, False) and not is_h.get(i1, False):
            h_count_map[i1] += 1

    G = nx.Graph()

    if explicit_h:
        # keep all atoms with original ordering for node ids (1 to N)
        for aid in sorted(atoms_tmp):
            a = atoms_tmp[aid]
            atom_key = (a["element"] or "?", a["sybyl_type"])
            G.add_node(
                aid,
                name=a["name"], element=a["element"], sybyl_type=a["sybyl_type"],
                x=a["x"], y=a["y"], z=a["z"], charge=a["charge"],
                subst_id=a["subst_id"], subst_name=a["subst_name"],
                original_index=a["original_index"],
                # "compat" attr
                atom_key=atom_key
            )
        for i1, j1, raw in bonds_tmp:
            order, aromatic = _bond_order_and_flags(raw)
            G.add_edge(
                i1, j1,
                order=order, aromatic=aromatic,
                raw_type=raw, bond_key=raw,
                has_h=is_h.get(i1, False) or is_h.get(j1, False),
                original_indices=(i1, j1)
            )

    else:
        # collapse hydrogens: only heavy atoms become nodes, reindexed 1..N
        for aid in heavy_ids:
            a = atoms_tmp[aid]
            hc = h_count_map.get(aid, 0)
            atom_key = (a["element"] or "?", a["sybyl_type"], hc)
            new_id = heavy_reindex[aid]
            G.add_node(
                new_id,
                name=a["name"], element=a["element"], sybyl_type=a["sybyl_type"],
                x=a["x"], y=a["y"], z=a["z"], charge=a["charge"],
                subst_id=a["subst_id"], subst_name=a["subst_name"],
                h_count=hc, original_index=aid,
                atom_key=atom_key
            )
        for i1, j1, raw in bonds_tmp:
            if is_h.get(i1, False) or is_h.get(j1, False):
                continue
            i_new = heavy_reindex[i1]; j_new = heavy_reindex[j1]
            order, aromatic = _bond_order_and_flags(raw)
            G.add_edge(
                i_new, j_new,
                order=order, aromatic=aromatic,
                raw_type=raw, bond_key=raw,
                has_h=False,
                original_indices=(i1, j1)
            )

    G.graph["explicit_h"] = explicit_h
    G.graph["heavy_reindex"] = heavy_reindex if not explicit_h else None
    return G

def find_nonzero_rows(csr_matrix, column_x):
    col_data = csr_matrix.getcol(column_x).toarray().flatten()
    nz_rows = np.nonzero(col_data)[0]
    if nz_rows.any():
        return nz_rows
    else:
        raise ValueError(f"No non-zero rows found in column {column_x}.")

def draw_ss_by_id_nx(fps, list_of_mols, featurizer, ind, add_to_ind=0, with_text=False, line2=''):
    """
    :param fps:
    :param list_of_mols:
    :param featurizer:
    :param ind:
    :param with_text:
    :return:
    """

    # firstly, find index of the first molecule with the desired substructure present
    first_molecule = find_nonzero_rows(fps, ind)[0]
    # then collect the id (hash) of the specified substructure
    ss_id = featurizer.bit_ids_[ind + add_to_ind]
    ss_atom_inds = featurizer.bi_fit_[first_molecule][ss_id][0]
    H = list_of_mols[first_molecule].subgraph(ss_atom_inds)
    im = draw_mol_nxgraph(H, just_show=False)
    if with_text:
        draw = ImageDraw.Draw(im)
        width, height = im.size
        line1 = str(ss_id)
        bbox1 = draw.textbbox((0, 0), line1) # TODO fix normal fonts here
        bbox2 = draw.textbbox((0, 0), line2)
        text_width1 = bbox1[2] - bbox1[0]
        text_height1 = bbox1[3] - bbox1[1]
        text_width2 = bbox2[2] - bbox2[0]
        text_height2 = bbox2[3] - bbox2[1]
        margin = 10
        x1 = (width - text_width1) / 2
        y1 = height - text_height1 - text_height2 - margin * 2
        x2 = (width - text_width2) / 2
        y2 = height - text_height2 - margin
        draw.text((x1, y1), line1, fill=(0, 0, 0))  # White color
        draw.text((x2, y2), line2, fill=(0, 0, 0)) # TODO fix fonts here too (font=font)
    return im