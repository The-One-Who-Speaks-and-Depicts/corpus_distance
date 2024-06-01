"""
Clusterisation module contains algorithms that perform actual
split of lects into groups,
based on the results of distance measurements, conducted earlier.
"""
import logging
from os.path import isdir, dirname, realpath
from dataclasses import dataclass, field
from typing import Callable
from Bio.Phylo.BaseTree import Tree
from Bio.Phylo.TreeConstruction import _DistanceMatrix, DistanceTreeConstructor
from corpus_distance.clusterisation import utils

def get_tree(distance_matrix: _DistanceMatrix,
             classification_method: Callable = DistanceTreeConstructor().upgma
             ) -> Tree:
    """
    Takes a distance matrix, lect names and any kind of
    method that builds a Phylo object (by default,
    DistanceTreeConstructor().upgma()); for further details,
    see BioPython documentation

    Parameters:
        distance_matrix(_DistanceMatrix): a lower triangular matrix
        of distances within lect pairs
        lects(list[str]): names of 
        classification_method(Callable): a function that returns a Phylo object
        on the basis a given distance matrix in a lower triangular format
    Returns:
        tree(Tree): an acquired phylogenetic tree
    """
    tree = classification_method(distance_matrix)
    return tree

@dataclass
class ClusterisationParameters:
    """
    Clusterisation parameters contains the main information on 
    how to cluster given lects

    Parameters:
        lects(list[str]): names of lects
        classification_method(Callable): a function that returns a Phylo object
        on the basis a given distance matrix in a lower triangular format
        data_name(str): a name of dataset
        outgroup(str): a proposed outgroup
        metrics(str): a name of metrics, used for hybridisation
        store_path(str): a path to store data
    """
    lects: list[str] = field(default_factory=list)
    outgroup: str = "default_outgroup"
    data_name: str = "default_data_name"
    metrics: str = "default_metrics_name"
    classification_method: Callable = DistanceTreeConstructor().upgma
    store_path: str = dirname(realpath(__file__))

def clusterise_lects_from_distance_matrix(
        pairwise_distances: list[tuple[tuple[str,str], int|float]],
        clusterisation_parameters: ClusterisationParameters) -> None:
    """
    A function that takes acquired distances between lect pairs, and creates tree,
    required information about it, and visualisation

    Parameters:
        pairwise_distances(list[tuple[tuple[str,str], int|float]]): a 1d-array
        of tuples that contain lect pairs and distances between given lects
        clusterisation_parameters(ClusterisationParameters): parameters for clusterisation
    """
    if not isdir(clusterisation_parameters.store_path):
        raise ValueError("Directory does not exist")
    logging.info('Distances are %s', pairwise_distances)
    distance_matrix = utils.create_distance_matrix(pairwise_distances,
                                                   clusterisation_parameters.lects)
    logging.info('Distance matrix is %s', distance_matrix)
    tree = get_tree(distance_matrix,
                    clusterisation_parameters.classification_method)
    logging.info('Tree is %s', tree)
    utils.detect_outgroup(tree,
                          clusterisation_parameters.outgroup,
                          clusterisation_parameters.data_name,
                          clusterisation_parameters.metrics,
                          clusterisation_parameters.store_path)
    utils.visualise_tree(tree,
                         clusterisation_parameters.metrics,
                         clusterisation_parameters.data_name,
                         clusterisation_parameters.store_path)
