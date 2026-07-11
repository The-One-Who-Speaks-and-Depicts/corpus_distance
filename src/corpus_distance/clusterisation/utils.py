"""
Clusterisation utils module contain functions that aid in clusterisation
by preparing existing data for the clustering functions.
"""

from os.path import dirname, isdir, join, realpath
import re

from Bio import Phylo


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
    is_outgroup_correct = 'CORRECT'\
        if outgroup in [tree.clade.clades[0].name, tree.clade.clades[1].name]\
        else 'INCORRECT'
    outgroup_clade = 1 if re.search(r'Inner\d{1,}', tree.clade.clades[0].name) else 0
    ingroup_clade = 0 if re.search(r'Inner\d{1,}', tree.clade.clades[0].name) else 1
    with open(join(store_path, metrics + "_clusterisation.info"), 'w', encoding='utf-8') as out:
        out.write(f"{data_name}\t{is_outgroup_correct}\t\
                  {tree.clade.clades[outgroup_clade].branch_length}\t\
                    {tree.clade.clades[ingroup_clade].branch_length}")
    Phylo.write(tree, join(store_path, metrics + ".newick"), 'newick')