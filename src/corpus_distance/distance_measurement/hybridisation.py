"""
Hybridisation module joins string similarity measures
and frequency metrics and uses them for scoring distances
between two lects.
"""
from dataclasses import dataclass
from typing import Callable
from tqdm import tqdm
from corpus_distance.distance_measurement.string_similarity\
    import jaro_vector_wrapper, StringSimilarityMeasurementInfo
from corpus_distance.distance_measurement.frequency_metrics\
    import dist_rank



@dataclass
class LectPairInformation:
    """
    A class that contains information on the pair of lects,
    necessary for comparison

    Parameters:
        lect_a_by_n_grams (list[(str, int | float)]): list of n-grams
        and their frequencies for the first lect
        lect_b_by_n_grams (list[(str, int | float)]): list of n-grams
        and their frequencies for the second lect
        lect_a_vectors (dict[str, list [int|float]]): dictionary
        with symbols as keys and their vectors within n-grams
        as values for the first lect
        lect_b_vectors (dict[str, list [int|float]]): dictionary
        with symbols as keys and their vectors within n-grams
        as values for the first lect
        lect_a_info (float): enthropy of the first lect alphabet
        lect_b_info (float): enthropy of the second lect alphabet
    """
    lect_a_n_grams: list[(str, int | float)]
    lect_b_n_grams: list[(str, int | float)]
    lect_a_vectors: dict[str, list [int|float]]
    lect_b_vectors: dict[str, list [int|float]]
    lect_a_info: float
    lect_b_info: float

def hybrid_for_single_lect(lects_information: LectPairInformation,
                           string_similarity_measure=jaro_vector_wrapper,
                           alphabet_normalisation = False
                           ) -> dict:
    """
    An auxiliary function that scores differences 
    between non-coinciding n-grams of a single lect 
    to their potential counterparts of the second lect

    Parameters:
        lects_information(LectPairInformation): a information
        on n-gram frequencies, symbol vectors and alphabet entropy
        for each of the lects
        string_similarity_measure(Callable): a string similarity
        measure for hybridisation
        alphabet_normalisation(bool): the option to use normalisation
        by alphabet entropy for vector-based measurements

    Returns:
        diffs_with_b(dict): a dictionary with lect a n-gram as keys
        and their differences to the closest non-coinciding 
        n-grams of lect b, as well as these n-grams as values
    """
    if (
        not lects_information.lect_a_n_grams\
        or not lects_information.lect_b_n_grams
        ):
        raise ValueError("There is no information on frequencies")
    if (
        not lects_information.lect_a_vectors\
        or not lects_information.lect_b_vectors
        ):
        raise ValueError("There is no information on vectors")
    if (
        not lects_information.lect_a_info\
        or not lects_information.lect_b_info
        ):
        raise ValueError("There is no information on entropy")
    diffs_with_b = {}
    for a in tqdm(lects_information.lect_a_n_grams):
        differences_for_a_single_n_gram = []
        for b in lects_information.lect_b_n_grams:
            # the first stage is two score a string similarity measure
            differences_for_a_single_n_gram.append(
                                    (b[0], abs(a[1] - b[1]),
                                    string_similarity_measure(
                                        StringSimilarityMeasurementInfo(
                                        str1 = a[0], vec1 = lects_information.lect_a_vectors,
                                        ent1 = lects_information.lect_a_info, str2 = b[0],
                                        vec2 = lects_information.lect_b_vectors,
                                        ent2 = lects_information.lect_b_info,
                                        alphabet_normalisation = alphabet_normalisation
                                        ))
                                    /max(len(a[0]), len(b[0]))))
            # the next stage is two score the minimal value
            # of string similarity measure,
            # the closest possible n-gram difference between
            # n-gram of lect a and given n-gram of lect b
            min_diff = min(
               list((i[2] for i in differences_for_a_single_n_gram))
               )
            # save only such n-grams of b
            # the distance to which matches minimal one
            b_n_grams_with_min_diff =\
                [i for i in differences_for_a_single_n_gram\
                if i[2] == min_diff]
            # score hybrid measure and transform the result for output
            b_n_grams_with_min_diff = [(
               i[0], (1 - ((1 - i[1]) * (1 - min_diff)))
               ) for i in b_n_grams_with_min_diff]
            diffs_with_b[a[0]] = b_n_grams_with_min_diff
    return diffs_with_b


def hybrid(lects_information: LectPairInformation,
        metrics=jaro_vector_wrapper,
        alphabet_normalisation = False
        ) -> tuple[list[int|float], int|float, tuple[dict, dict]]:
    """
    A metric that scores hybrid measurements between two lects

    Parameters:
        lects_information(LectPairInformation): a information
        on n-gram frequencies, symbol vectors and alphabet entropy
        for each of the lects
        string_similarity_measure(Callable): a string similarity
        measure for hybridisation
        alphabet_normalisation(bool): the option to use normalisation
        by alphabet entropy for vector-based measurements
    Returns: 
        results(tuple):
            hybrid_results(list[int|float]): a list of metric 
            values for each n-gram
            mean_value(int|float): a mean normalised value
            of metric values
            diffs_to_output(tuple[dict, dict]): two dicts 
            for n-grams of lects a and b respectively, each
            containing n-gram of lect, the one that is closest to
            it in the other, and value of their distances
    """
    if (
        not lects_information.lect_a_n_grams\
        or not lects_information.lect_b_n_grams
        ):
        raise ValueError("There is no information on frequencies")
    if (
        not lects_information.lect_a_vectors\
        or not lects_information.lect_b_vectors
        ):
        raise ValueError("There is no information on vectors")
    if (
        not lects_information.lect_a_info\
        or not lects_information.lect_b_info
        ):
        raise ValueError("There is no information on entropy")
    # performing an operation for each lect
    diffs_a = hybrid_for_single_lect(lects_information,
                                    metrics,
                                    alphabet_normalisation)
    flipped_lect = LectPairInformation(
       lect_a_n_grams=lects_information.lect_b_n_grams,
       lect_b_n_grams=lects_information.lect_a_n_grams,
       lect_a_vectors=lects_information.lect_b_vectors,
       lect_b_vectors=lects_information.lect_a_vectors,
       lect_a_info=lects_information.lect_b_info,
       lect_b_info=lects_information.lect_a_info
    )
    diffs_b = hybrid_for_single_lect(flipped_lect,
                                    metrics,
                                    alphabet_normalisation)
    # create a tuple for displaying contrasted n-grams afterwards
    diffs_to_output = (diffs_a, diffs_b)
    # get metric values together
    hybrid_joined_n_grams =\
    [i[1] for j in\
     [list(diffs_a.items()), list(diffs_b.items())] for i in j\
    ]
    hybrid_results =\
        [i[1] for j in hybrid_joined_n_grams for i in j]
    return (
       hybrid_results,
       sum(hybrid_results)/len(hybrid_results),
       diffs_to_output
    )


@dataclass
class HybridisationParameters:
    """
    A class that contains configuration of the hybrid metrics.

    Parameters:
        hybridisation(bool): an option to use string similarity
        measure for hybridisation with DistRank
        soerensen(bool): flag that allows for normalisation
        based on multiplying resulting value
        by Soerensen-Dice coefficient
        hybridisation_as_array(bool): an option to use results from
        DistRank and string similarity measures as two arrays, and
        not two separate values
        metrics(Callable): a string similarity
        measure for hybridisation
        alphabet_normalisation(bool): the option to use normalisation
        by alphabet entropy for vector-based measurements 
    """
    hybridisation: bool = True
    soerensen: bool = True
    hybridisation_as_array: bool = False
    metrics: Callable = jaro_vector_wrapper
    alphabet_normalisation: bool = True

    def __str__(self):
        """
        Returns description of hybridisation parameters in the following order:
        * Introducing soerensen coefficient
        * Introducing hybridisation
        * Taking all the metric values as a single array
        * string similarity measure, used for hybridisation
        * normalisation by alphabet entropy for vector-based measures only
        """
        string_similarity_measure_name = self.metrics.__name__ if self.metrics.__name__ not in [
            'jaro_vector_wrapper', 'vector_measure_wrapper'
            ] else f'{self.metrics.__name__}-{self.alphabet_normalisation}'
        return f'{self.soerensen}-\
            {self.hybridisation}-{self.hybridisation_as_array}-{string_similarity_measure_name}'



def compare_lects_with_vectors(lects_information: LectPairInformation,
    hybridisation_parameters: HybridisationParameters = HybridisationParameters()
    ) -> tuple[tuple[dict, dict], int|float]:
    """
    A function that takes information on two lects and parameters for hybridisation
    in order to perform computation of a joined metrics

    Parameters:
        lects_information(LectPairInformation): information on pair of lects to compare
        hybridisation_parameters(HybridisationParameters): settings for hybridisation
    Returns:
        results(tuple[dict|dict]):
            dist_rank_for_analysis(dict): an output for DistRank function
                list_of_ranks(list[int|float]): list of metric values
                mean_of_ranks(int|float): mean metric value
                n_grams_for_analysis(list[(str, int|float)]): n-grams with information
                on metric value between them
            hybrid_for_analysis(dict): an output for hybrid function:
                list_of_ranks(list[int|float]): list of metric values
                mean_of_ranks(int|float): mean metric value
                n_grams_for_analysis(list[(str, int|float)]): n-grams with information
                on metric value between them
        hybrid_value (int|float): a mean between hybrid function and distrank
    """
    # save coinciding and non-coinciding n-grams
    coinciding_n_grams_number = len(
        [i for i in lects_information.lect_a_n_grams if i[0] in\
                          [j[0] for j in lects_information.lect_b_n_grams]]
    )
    non_coinciding_n_grams = LectPairInformation(
        lect_a_n_grams=[i for i in lects_information.lect_a_n_grams if i[0] not in\
                             [j[0] for j in lects_information.lect_b_n_grams]
                             ],
        lect_b_n_grams=[i for i in lects_information.lect_b_n_grams if i[0] not in\
                              [j[0] for j in lects_information.lect_a_n_grams]
                            ],
        lect_a_vectors=lects_information.lect_a_vectors,
        lect_b_vectors=lects_information.lect_b_vectors,
        lect_a_info=lects_information.lect_a_info,
        lect_b_info=lects_information.lect_b_info
    )
    # scoring DistRank
    dist_ranks, dist_res, dist_rank_for_analysis = dist_rank(coinciding_n_grams_number,
        lects_information.lect_a_n_grams, lects_information.lect_b_n_grams,
        hybridisation_parameters.soerensen)
    if hybridisation_parameters.hybridisation:
    # scoring hybrid measurements
        hybrid_ranks, hybrid_res, hybrid_for_analysis = hybrid(
                non_coinciding_n_grams,
                hybridisation_parameters.metrics,
                hybridisation_parameters.alphabet_normalisation
                )
        # if the choice is to join all the results into a single array
        if hybridisation_parameters.hybridisation_as_array:
            # it is necessary to check, whether there is a single coincidence,
            # to escape potential errors
            if len(dist_ranks) > 0:
                # join measurements into a single array
                # and normalise the distribution,
                # in order that anomalies in frequency/similarity
                # distribution would not affect the result
                overall_scores = [i for j in [hybrid_ranks, dist_ranks] for i in j]
                # get the mean of distribution and
                # return it across with mean of all metric values
                return (
                    dist_rank_for_analysis, hybrid_for_analysis
                    ), sum(overall_scores)/len(overall_scores)
            # if there are no coincidence, use only non-coinciding n-gram measurements
            return ({}, hybrid_for_analysis), hybrid_res
        # otherwise, DistRank and hybrid metrics get the completely equal treatment
        # it is necessary to check, whether there is a single coincidence,
        # to escape potential errors
        if len(dist_ranks) > 0:
            # if there are coincidences, return all the n-grams and
            # normalised measure
            return (
                dist_rank_for_analysis, hybrid_for_analysis
                ), dist_res * hybrid_res
        # otherwise, return only hybrid metrics
        return ({}, hybrid_for_analysis), hybrid_res
    if len(dist_ranks) == 0:
        # if there is not a single DistRank, throw error: there is nothing to
        # measure without hybrid metrics chosen
        raise ValueError('No coinciding n-grams for pure DistRank')
    # otherwise, return only DistRanks
    return (dist_rank_for_analysis, {}), dist_res
