from os.path import join

from matplotlib import rc
from matplotlib.pyplot import Figure, figure, ylabel, close, axis, savefig, tight_layout


import networkx as nx
from Bio.Phylo import draw
from Bio.Phylo.BaseTree import Tree

from corpus_distance.clusterisation.clusterisation import NetworkParams


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

def visualise_network(params: NetworkParams,
                      out_path,
                      metrics_name: str,
                      node_size=10, font_size=7,
                      dpi=300,
                      widths: float=0.8,
                      ) -> Figure:
    
    fig = figure(figsize=(8, 8), dpi=dpi)

    nx.draw_networkx_edges(params.graph, params.pos, width=widths, edge_color="black", alpha=0.9)
    nx.draw_networkx_nodes(params.graph, params.pos, nodelist=params.leaves, node_size=node_size, node_color="black")

    leaf_labels = {n: params.labels.get(n, str(n)) for n in params.leaves}
    nx.draw_networkx_labels(params.graph, params.pos, labels=leaf_labels, font_size=font_size)

    axis("equal"); axis("off"); tight_layout(pad=0.02)
    savefig(join(out_path, "nn_" + metrics_name + ".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    savefig(join(out_path, "nn_" + metrics_name + ".svg"), bbox_inches="tight", pad_inches=0.01)
    close()
    return fig