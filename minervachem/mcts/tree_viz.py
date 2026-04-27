import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def make_tree_nodes(node, size:int):
    """Creates the edges for plotting in a tree diagram. The nodes need a unique identifier ("_int") to be graphed

    Args:
        node (Node class): the final node from the MCTS search, contains all information on the children
        size (int): the number of turns from the MCTS search

    Returns:
        tuple: creates tuples of nodes & edges
    """
    paths = []
    tuples = []

    for _ in range(size):
        if node.parent:
            node = node.parent
        else:
            break

        if node:
            children = node.children
            for i in children:
                moves = i.state.moves
                paths.append(moves)

    for j in range(size):
        level = [p for p in paths if len(p) == j+1]
        # print("Levels", level)
        for partial_mol in level:
            if len(partial_mol) > 1:
                second_last, last = partial_mol[-2:]
                last = last.replace("\n",'.')
                second_last = second_last.replace("\n", ".")
                tuple = (f"{second_last}_{j}", f"{last}_{j+1}")
                # e.g. ("C_4","S_5") if the string is CCCCS
            else:
                last = partial_mol[-1].replace("\n", ".")
                tuple = ('&',  f"{last}_{j+1}")

            tuples.append(tuple)

    return tuples

def make_node_info(node, size:int):
    """Make a dictionary of all the attributes of a node's children.

    Args:
        node (Node): node
        size (int): max level of tree search

    Returns:
        dict: dictionary of information on the node's state
    """

    node_info = {}
    for j in range(size):

        if node.parent:
            node = node.parent
        else:
            break

        if node:
            children = node.children
            for i in children:
                key = str(i.state.moves[-1]).replace("\n", ".") + "_"+ str(size-j)
                dict_keys = i.state.__dict__.keys()
                row_dict = {}
                for k in dict_keys:
                    row_dict[k] = getattr(i.state, k) if getattr(i.state, k) is not None else np.nan
                    row_dict['visits'] = i.visits if i.visits is not None else np.nan
                    row_dict['reward'] = i.state.reward() if i.state.reward() is not None else np.nan
                    row_dict['turn'] = size - j
                    row_dict['label'] = str(i.state.moves[-1]).replace("\n", ".")
                node_info[key] = row_dict

    root_dict = {i: np.nan for i in node.children[0].state.__dict__.keys()}
    add = {'turn': 0, 'label': 'root', 'visits': np.nan, 'reward': np.nan}
    combined = {**root_dict, **add}
    node_info['&'] = combined
    return node_info


def graph_tree(param:str, tuples:list, node_info:dict, smiles:str, cmap):
    """Use networkx to visualize a search tree for a given node attribute value.

    Args:
        param (str): label for value of statue being plotted
        tuples (list): list of node edges
        node_info (dict): dictionary of information on the node's state
        smiles (str): smiles string
        cmap (_type_): color map for plotting
    """

    # make a big 'ol plot
    fig, ax = plt.subplots(figsize=(15,15))
    G=nx.Graph()

    # add all the attributes from the node_info dict to the nodes being graphed
    for node, attr in node_info.items():
        G.add_node(node, **attr)

    # add node edges
    G.add_edges_from(tuples)

    node_color_values = [attr[param] for node, attr in G.nodes(data=True)]

    vmin = min(node_color_values)
    vmax = max(node_color_values)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = cmap

    node_colors = [cmap(norm(value)) if not np.isnan(value) else 'lightgray' for value in node_color_values]

    pos = hierarchy_pos(G,'&', width=20, height=20)
    nx.draw(
        G, 
        pos=pos, 
        node_color=node_colors, 
        with_labels=True, 
        cmap=cmap, 
        font_size = 14, 
        node_size=1000, 
        labels={node: attr['label'] for node, attr in G.nodes(data=True)}
        )

    for node, (x, y) in pos.items():
        if isinstance(G.nodes[node][param], float):
            label = f"{param}:\n{G.nodes[node][param]:.3f}"
        else:
            label = f"{param}:\n{G.nodes[node][param]}"
        ax.text(x, y - 0.4, label, ha='center', va='top', fontsize=12, wrap=True)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ax = plt.gca()
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(f'{param}', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    clean_smiles = smiles.replace("\n", ".")
    plt.title(f"{param} of '{clean_smiles}' Search Tree", fontsize=24)

    plt.show()


def hierarchy_pos(G, root, levels=None, width=1., height=4.):
    """
    from: https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3/29597209#29597209

    If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node
       levels: a dictionary
               key: level number (starting from 0)
               value: number of nodes in this level
       width: horizontal space allocated for drawing
       height: vertical space allocated for drawing
    """
    TOTAL = "total"
    CURRENT = "current"
    def make_levels(levels, node=root, current_level=0, parent=None):
        """Compute the number of nodes for each level
        """
        if not current_level in levels:
            levels[current_level] = {TOTAL : 0, CURRENT : 0}
        levels[current_level][TOTAL] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                levels =  make_levels(levels, neighbor, current_level + 1, node)
        return levels

    def make_pos(pos, node=root, current_level=0, parent=None, vert_loc=0):
        dx = 1/levels[current_level][TOTAL]
        left = dx/2
        pos[node] = ((left + dx * levels[current_level][CURRENT]) * width, vert_loc)
        levels[current_level][CURRENT] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                pos = make_pos(pos, neighbor, current_level + 1, node, vert_loc - vert_gap)
        return pos
    if levels is None:
        levels = make_levels({})
    else:
        levels = {l:{TOTAL: levels[l], CURRENT:0} for l in levels}
    vert_gap = height / (max([l for l in levels])+1)
    return make_pos({})


def mass_plotting(node_info:dict, params:list, tuples:list, smiles:str):
    """Mass tree plotting.

    Args:
        node_info (dict): dictionary of information on the node's state
        params (list): list of labels of node attributes to be plotted
        tuples (list): list of node edges
        smiles (str): smiles string
    """
    maps = [
        'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
    ]

    for i, j in enumerate(params):
        cmap = getattr(cm, maps[i])
        graph_tree(param=j, tuples=tuples, node_info=node_info, smiles=smiles, cmap=cmap)

