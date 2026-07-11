from typing import NamedTuple

from numpy import number

from corpus_distance.cdutils import LectPair

class LectPairDistance(NamedTuple):
    lect_names: LectPair
    distance: number


def create_matrix(pairwise_distances: LectPairDistance,
                  lects: list[str], diagonal: bool = True) ->  list[list[number]]:
    
    final_matrix = []
    for i in range(len(lects)):
        final_matrix.append([])
        for j in range(len(lects)):
            if j < i:
                dist =\
                    [d[1] for d in pairwise_distances if\
                        set([lects[i], lects[j]]) == set([d[0][0], d[0][1]])][0]
                final_matrix[i].append(dist)
        final_matrix[i].append(0)
        if not diagonal:
            for j in range(len(lects)):
                if j > i:
                    dist =\
                        [d[1] for d in pairwise_distances if\
                            set([lects[i], lects[j]]) == set([d[0][0], d[0][1]])][0]
                    final_matrix[i].append(dist)
    return final_matrix