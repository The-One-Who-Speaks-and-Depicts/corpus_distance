from os.path import join

from matplotlib import rc
from matplotlib.pyplot import Figure, figure, ylabel, close, axis, savefig, tight_layout

from pandas import DataFrame

import networkx as nx
import fastnntpy as fn
from Bio.Phylo import draw
from Bio.Phylo.BaseTree import Tree


def visualise_tree(tree: Tree, metrics: str, data_name: str,
                   ) -> Figure:
    """
    Visualises tree with matplotlib.

    Parameters:
        tree(Phylo.BaseTree.Tree): BioPython tree for visualisation
        metrics(str): name of metrics, with which tree was built
        data_name(str): name of data, for which tree was built
    
    Returns:
        plt.Figure: a plotted tree
    """
    if not isinstance(tree, Tree):
        raise ValueError(f"tree should be a Phylo.BaseTree.Tree, received {tree}")
    if not metrics or not isinstance(metrics, str):
        raise ValueError(f"metrics should be a non-empty string, received {metrics}")
    if not data_name or not isinstance(data_name, str):
        raise ValueError(f"data_name should be a non-empty string, received {data_name}")
    font = {'family':'DejaVu Sans', 'weight':'normal', 'size':20}
    rc('font', **font)
    fig = figure(figsize=(40, 15))
    fig_title = f'{metrics} of {data_name}' if data_name not in metrics else metrics
    fig.suptitle(fig_title, fontsize=36)
    axes = fig.add_subplot(1, 1, 1)
    axes.set(yticklabels=[],yticks=[])
    draw(tree, axes=axes, show_confidence=False, do_show=False)
    ylabel("")
    close()
    return fig

def visualise_network(matrix: list[list[int|float]],
                      lects: list[str],
                      out_path,
                      metrics_name: str,
                      shift=0, node_size=10, font_size=7,
                      scale_width_by_weight=False, dpi=300,
                      ) -> None:
    data_frame = DataFrame(matrix, columns=lects)
    nx_obj = fn.run_neighbour_net(data_frame)
    # -- data from PyNexus --
    labels = {i + shift: s for i, s in nx_obj.get_node_translations()}
    pos    = {i + shift: (x, y) for i, x, y in nx_obj.get_node_positions()}
    # corrected parsing order: (edge_id, u, v, sid, w)
    edges_raw = [ (u + shift, v + shift, w)
                  for (_, u, v, _, w) in nx_obj.get_graph_edges() ]

    # only keep edges whose endpoints have positions
    edges = [(u, v, w) for (u, v, w) in edges_raw if u in pos and v in pos]
    if not edges:
        raise ValueError("No drawable edges (endpoints missing positions).")

    # -- build graph --
    G = nx.Graph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    # leaves only (degree == 1)
    leaves = [n for n, d in G.degree() if d == 1]

    # edge widths (optional)
    if scale_width_by_weight:
        ws = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
        wmax = max(ws) if ws else 1.0
        widths = [0.5 + 2.5 * (w / wmax) for w in ws]
    else:
        widths = 0.8

    # -- draw (no layout) --
    figure(figsize=(8, 8), dpi=dpi)

    nx.draw_networkx_edges(G, pos, width=widths, edge_color="black", alpha=0.9)
    nx.draw_networkx_nodes(G, pos, nodelist=leaves, node_size=node_size, node_color="black")

    leaf_labels = {n: labels.get(n, str(n)) for n in leaves}
    nx.draw_networkx_labels(G, pos, labels=leaf_labels, font_size=font_size)

    axis("equal"); axis("off"); tight_layout(pad=0.02)
    savefig(join(out_path, "nn_" + metrics_name + ".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    savefig(join(out_path, "nn_" + metrics_name + ".svg"), bbox_inches="tight", pad_inches=0.01)
    close()