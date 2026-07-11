from matplotlib.pyplot import rcParams
from seaborn import clustermap
from seaborn.matrix import ClusterGrid
from numpy import issubdtype, number
from pandas import DataFrame
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

def plot_clustermap(matrix: list[list[int|float]], lects: list[str]) -> ClusterGrid:
    if not lects or not isinstance(
        lects, list
    ) or not all(
        isinstance(x, str) and x.strip() for x in lects
        ):
        raise ValueError(f"lects should be a list of non-empty strings, received {lects}")
    if matrix is None or not isinstance(
        matrix,
        list
    ) or not all(        
        isinstance(row, list) and all(
            issubdtype(type(val), number) for val in row
            ) and len(row) == len(matrix) and len(row) == len(matrix[0])
            for row in matrix
    ):
        raise ValueError("matrix should be a list of numerical lists "\
                         "that are equal in length between themselves and with the matrix, "\
                         f"received {matrix}")
    df_res = DataFrame(matrix, index=lects, columns=lects)
    rcParams.update({'font.size': 30})
    condensed_distances = squareform(df_res.values)
    linkage_matrix = linkage(condensed_distances, method='average')
    
    res = clustermap(
        df_res,
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix,
        annot=True,
        figsize=(20,20)
    )
    return res