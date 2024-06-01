"""
Clusterisation utils module contain functions that aid in clusterisation
by preparing existing data for the clustering functions.
"""

from os.path import dirname, isdir, join, realpath
import matplotlib
import matplotlib.pyplot as plt
from Bio import Phylo
from Bio.Phylo.TreeConstruction import _DistanceMatrix

def create_distance_matrix(
        pairwise_distances: list[tuple[tuple[str, str], int|float]],
        lects: list[str]) -> list[list[int|float]]:
    """
    Takes list of distances between pairs of lects and names of lects,
    and returns lower triangular distance matrix.

    Parameters:
        pairwise_distances(list[tuple[tuple[str, str], int|float]]): a 1d-array of distances
        between given lects with lect names
        lects(list[str]): names of lects
    Returns:
        distance_matrix(_DistanceMatrix): a DistanceMatrix between given lects
    """
    final_matrix = []
    ordered_lects = list(set(lects))
    for i in range(len(ordered_lects)):
        final_matrix.append([])
        for j in range(len(ordered_lects)):
            if j < i:
                dist =\
                    [d[1] for d in pairwise_distances if\
                        set([ordered_lects[i], ordered_lects[j]]) == set([d[0][0], d[0][1]])][0]
                final_matrix[i].append(dist)
        final_matrix[i].append(0)
    distance_matrix = _DistanceMatrix(ordered_lects, final_matrix)
    return distance_matrix

def detect_outgroup(tree: Phylo.BaseTree.Tree, outgroup: str, data_name: str,
                    metrics: str, store_path: str = dirname(realpath(__file__))) -> None:
    """
    Takes a rooted tree and prints, whether the first split clade is correct.

    Parameters:
        tree(Tree): a phylogenetic tree of Bio.Phylo.BaseTree.Tree class
        data_name(str): a name of dataset
        outgroup(str): a proposed outgroup
        metrics(str): a name of metrics, used for hybridisation
        store_path(str): a path to store data
    """
    if not isdir(store_path):
        raise ValueError("Directory not exists")
    if tree.rooted is False:
        raise ValueError("Tree is unrooted, not possible to detect outgroup")
    if (outgroup in [tree.clade.clades[0].name, tree.clade.clades[1].name]):
        outgroup_clade = 0 if tree.clade.clades[0].name == outgroup else 1
        ingroup_clade = 1 if tree.clade.clades[0].name == outgroup else 0
        with open(join(store_path, metrics + ".info"), 'w', encoding='utf-8') as out:
            out.write(f"{data_name}\tCORRECT\t{tree.clade.clades[outgroup_clade].branch_length}\t\
                      {tree.clade.clades[ingroup_clade].branch_length}")
    else:
        with open(join(store_path, metrics + ".info"), 'w', encoding='utf-8') as out:
            out.write(f"{data_name}\tINCORRECT\tNA\tNA")
    Phylo.write(tree, join(store_path, metrics + ".newick"), 'newick')


def visualise_tree(tree: Phylo.BaseTree.Tree, metrics: str, data_name: str,
                   store_path: str = dirname(realpath(__file__))) -> None:
    """
    Visualises tree with matplotlib.

    Parameters:
        tree(Phylo.BaseTree.Tree): BioPython tree for visualisation
        metrics(str): name of metrics, with which tree was built
        data_name(str): name of data, for which tree was built
    """
    font = {'family':'DejaVu Sans', 'weight':'normal', 'size':20}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(40, 15))
    fig.suptitle(f'{metrics} of {data_name}', fontsize=36)
    axes = fig.add_subplot(1, 1, 1)
    Phylo.draw(tree, axes=axes, show_confidence=False, do_show=False)
    plt.savefig(join(store_path, "phylogeny_" + metrics + "_" + data_name + ".png"))
    plt.close()
