"""
Clusterisation module contains algorithms that perform actual
split of lects into groups,
based on the results of distance measurements, conducted earlier.
"""
from logging import getLogger, NullHandler
from dataclasses import dataclass
from typing import Callable
from pandas import DataFrame
from Bio.Phylo.BaseTree import Tree
from Bio.Phylo.TreeConstruction import _DistanceMatrix, DistanceTreeConstructor
from fastnntpy import run_neighbour_net, Nexus

import networkx as nx
logger = getLogger(__name__)
logger.addHandler(NullHandler())

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



def create_neigbour_net_graph(nx_obj: Nexus, pos: dict[int, tuple[float, float]], shift: int = 0) -> nx.Graph:
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
    return G
@dataclass
class NetworkParams():
    graph: nx.Graph
    labels: dict[int, str]
    pos: dict[int, tuple[float, float]]
    leaves: list[str]

def create_neighbour_net_params(matrix: list[list[int|float]], lects: list[str], shift: int = 0) -> NetworkParams:
    data_frame = DataFrame(matrix, columns=lects)

    nx_obj = run_neighbour_net(data_frame)
    
    pos    = {i + shift: (x, y) for i, x, y in nx_obj.get_node_positions()}
    graph = create_neigbour_net_graph(nx_obj, pos, shift)
    labels = {i + shift: s for i, s in nx_obj.get_node_translations()}
    
    network = NetworkParams(
        graph=graph,
        labels=labels,
        pos=pos,
        leaves=[n for n, d in graph.degree() if d == 1]
    )

    logger.debug("Result: %s", network)
    return network


