from logging import getLogger, NullHandler
from os.path import isdir, join, dirname, realpath
from dataclasses import dataclass, field
from typing import Callable
from Bio.Phylo.TreeConstruction import _DistanceMatrix, DistanceTreeConstructor
from corpus_distance.visualisation.clusterisation import (
#    visualise_network,
    visualise_tree
)
from corpus_distance.visualisation.distance_measurement import plot_clustermap
from corpus_distance.distance_measurement.utils import create_matrix
from corpus_distance.clusterisation.clusterisation import get_tree
from corpus_distance.clusterisation.utils import detect_outgroup

logger = getLogger(__name__)
logger.addHandler(NullHandler())

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
    logger.info('Distances are %s', pairwise_distances)
    matrix = create_matrix(
            pairwise_distances,
            clusterisation_parameters.lects,
            False
            )
    logger.info('Preliminary matrix is %s', matrix)
    heatmap = plot_clustermap(matrix, clusterisation_parameters.lects)
    heatmap_path = join(
        clusterisation_parameters.store_path,
        "heatmap_" + clusterisation_parameters.metrics + ".png"
    )
    logger.info('Storing clustermap visualisation in %s', heatmap_path)
    heatmap.savefig(heatmap_path)
    diag_matrix = create_matrix(pairwise_distances, clusterisation_parameters.lects)
    distance_matrix = _DistanceMatrix(
        clusterisation_parameters.lects,
            diag_matrix
            )
    logger.info('Distance matrix is %s', distance_matrix)
    
    tree = get_tree(distance_matrix,
                    clusterisation_parameters.classification_method)
    logger.info('Tree is %s', tree)
    detect_outgroup(tree,
                          clusterisation_parameters.outgroup,
                          clusterisation_parameters.data_name,
                          clusterisation_parameters.metrics,
                          clusterisation_parameters.store_path)
    tree_visualisation = visualise_tree(tree,
                         clusterisation_parameters.metrics,
                         clusterisation_parameters.data_name,
                         clusterisation_parameters.store_path)
    tree_visualisation_path = join(
        clusterisation_parameters.store_path,
        "phylogeny_" + clusterisation_parameters.metrics +\
        "_" + clusterisation_parameters.data_name + ".png"
    )
    logger.debug('Storing tree visualisation in %s', tree_visualisation_path)
    tree_visualisation.savefig(tree_visualisation_path)