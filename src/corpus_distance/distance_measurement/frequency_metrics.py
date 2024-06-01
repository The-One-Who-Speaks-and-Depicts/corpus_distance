"""
Frequency metrics module contains functions
that score metric values that compare distribution
of n-grams across the analysed lects for
further integration into the pipeline
"""

from tqdm import tqdm
import Levenshtein

def soerensen_dice(lect1: int, lect2: int, common: int) -> float:
    """
    Scoring Soerensen-Dice coefficient for two given lects. 
    Soerensen coefficient is useful for normalising 
    DistRank metric's contribution to the final value, 
    reducing it when the lects are similar, and increasing,
    when they are different

    Arguments:
        lect1(int): number of n-grams in the first lect
        lect2(int): number of n-grams in the second lect
        common(int): number of intersecting n-grams between
        two lects
    Returns:
        soerensen(float): Soerensen-Dice coefficient
        between two lects
    """
    if (lect1 < 1 or lect2 < 1):
        raise ValueError("One of provided lengths is 0")
    return 2*common/(lect1+lect2)



def dist_rank(coinciding_n_grams_number: int,
              lect_a: list[(str, int|float)],
              lect_b: list[(str, int|float)],
              soerensen:bool = False
              ) -> tuple[
                 list[int|float],
                 int|float,
                 list[(str, int|float)]
              ]:
    """
    DistRank is a mean of absolute differences between ranks
    of coinciding n-grams in lects a and b.

    Arguments:
        coinciding_n_grams_number(int): number of common
        n-grams between two lects
        lect_a (list[(str, int|float)]): n-grams with
        their frequencies for the first lect
        lect_b (list[(str, int|float)]): n-grams with
        their frequencies for the second lect
        soerensen(bool): flag that allows for normalisation
        based on multiplying resulting value
        by Soerensen-Dice coefficient

    Returns:
        (list_of_ranks, mean_of_ranks, n_grams_for_analysis) -
        list_of_ranks(list[int|float]): list of metric values
        mean_of_ranks(int|float): mean metric value
        n_grams_for_analysis(list[(str, int|float)]): n-grams with information
        on metric value between them
    """
    # in case there are no coinciding n-grams, returns
    # an empty result
    list_of_ranks = []
    mean_of_ranks = 0
    n_grams_for_analysis = {}
    if coinciding_n_grams_number < 1:
        return (list_of_ranks, mean_of_ranks, n_grams_for_analysis)
    for i in tqdm(lect_a):
        for j in lect_b:
            # if Levenshtein distance is 0, then n-grams coincide
            if Levenshtein.distance(i[0], j[0]) == 0:
                # score dist_rank
                list_of_ranks.append(abs(i[1] - j[1]))
                # add information
                n_grams_for_analysis[i[0]] = abs(i[1] - j[1])
    # yielding all the scores, and mean normalised DistRank score
    mean_of_ranks = sum(list_of_ranks)/len(list_of_ranks)
    # if Soerensen normalisation utilised, change result accordingly
    if soerensen:
        soerensen_coefficient =\
            soerensen_dice(len(lect_a), len(lect_b), coinciding_n_grams_number)
        mean_of_ranks = mean_of_ranks/soerensen_coefficient
    return (list_of_ranks, mean_of_ranks, n_grams_for_analysis)
