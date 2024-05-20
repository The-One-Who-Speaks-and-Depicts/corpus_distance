"""
String similarity module provides
wrappings for different string similarity measures,
in order for them to be sufficiently
integrated into the pipeline.
"""
from dataclasses import dataclass, field
import Levenshtein
from scipy.spatial.distance import euclidean
from pyjarowinkler.distance import get_jaro_distance as jw



@dataclass
class StringSimilarityMeasurementInfo:
    """
    Contains information on possible set of parameters
    that string similarity measures utilise.

    Arguments:
        str1(str): symbol sequence from lect 1
        vec1(dict): dictionary for symbol vectors
         from lect 1
        ent1(dict): alphabet entropy for lect 1
        str2(str): symbol sequence from lect 2
        vec2(dict): dictionary for symbol vectors
         from lect 2
        ent2(dict): alphabet entropy for lect 2
        alphabet_normalisation(bool): flag to 
        perform normalisation by alphabet entropy 
        (used for vector-based
         string similarity measures)
    """
    str1: str = ""
    vec1: dict = field (default_factory=dict)
    ent1: float = 0
    str2: str = ""
    vec2: dict = field (default_factory=dict)
    ent2: float = 0
    alphabet_normalisation: bool = True


def vector_measure_wrapper(
      strings_info: StringSimilarityMeasurementInfo) -> float:
    """
    Takes two strings, the symbol vectors from their lects,
    alphabet entropy information,
    and flag for alphabet normalisation. 
    Scores Euclidean distance between 
    sum of vectors for each symbol within the given string. 
    Afterwards multiplied by division of smaller entropy
    by bigger entropy, if alphabet normalisation flag
    is set to True. Finally, split by length of 
    the longer sequence.

    Arguments:
        strings_info(StringSimilarityMeasurementInfo):
        an object with required parameters: str1, str2,
        vec1, vec2, ent1, ent2; and optional patameter
        alphabet_normalisation. For details, see 
        StringSimilarityMeasurementInfo description
    Returns:
        distance(float): distance between given sequences with
        vectors and alphabet entropy taken into
        consideration
    """
    if (not strings_info.str1 or not strings_info.str2):
        raise ValueError("One of strings is absent")
    if (not strings_info.vec1 or not strings_info.vec2):
        raise ValueError("One of vector dictionaries is absent")
    if ((not strings_info.ent1 or not strings_info.ent1) \
        and strings_info.alphabet_normalisation):
        raise ValueError("There is no alphabet entropy information")
    sum_s1 = sum(
    list(strings_info.vec1[v] for _, v in list(
    enumerate(strings_info.str1
                ))))
    sum_s2 = sum(
    list(strings_info.vec2[v] for _, v in list(
    enumerate(strings_info.str2
                ))))
    diff = euclidean(sum_s1, sum_s2)
    alphabet_difference = \
    strings_info.ent1/strings_info.ent2 \
        if strings_info.ent1 > strings_info.ent2 \
            else strings_info.ent2/strings_info.ent1
    length_normalisation = max([len(strings_info.str1), len(strings_info.str2)])
    return diff/length_normalisation \
        if not strings_info.alphabet_normalisation \
            else (diff * alphabet_difference)/length_normalisation


def levenshtein_wrapper(strings_info: StringSimilarityMeasurementInfo) -> float:
    """
    Takes two strings and scores Levenshtein distance
    (number of additions, deletions, and substitutions) between them.
    Divides the result but length of the longer string.

    Arguments:
        strings_info(StringSimilarityMeasurementInfo):
        an object with required parameters: str1 and str2. 
        For details, see StringSimilarityMeasurementInfo description
    
    Returns:
        distance(float): Levenshtein distance normalised
        between two given strings
    """
    if (not strings_info.str1 or not strings_info.str2):
        raise ValueError("One of strings is absent")
    return Levenshtein.distance(strings_info.str1, strings_info.str2)/\
    max([len(strings_info.str1), len(strings_info.str2)])



def weighted_jaro_winkler_wrapper(
        strings_info: StringSimilarityMeasurementInfo) -> float:
    """
    Takes two strings and scores Jaro-Winkler distance
    (number of transpositions, multiplied by coefficient
    that fines strings with dissimilar beginnings) between them.
    Subtracts the result from 1. Multiplies the result
    by Levenshtein distance between two strings.
    Divides the result but length of the longer string.

    Arguments:
        strings_info(StringSimilarityMeasurementInfo):
        an object with required parameters: str1 and str2. 
        For details, see StringSimilarityMeasurementInfo description
    
    Returns:
        distance(float): Levenshtein distance normalised
        between two given strings
    """
    if (not strings_info.str1 or not strings_info.str2):
        raise ValueError("One of strings is absent")
    levenshtein = Levenshtein.distance(strings_info.str1, strings_info.str2)/\
        max([len(strings_info.str1), len(strings_info.str2)])
    jaro_winkler = jw(strings_info.str1, strings_info.str2)/\
        max([len(strings_info.str1), len(strings_info.str2)])
    return levenshtein * (1 - jaro_winkler)



def jaro_vector_wrapper(strings_info: StringSimilarityMeasurementInfo) -> float:
    """
    Takes two strings, the symbol vectors from their lects,
    alphabet entropy information,
    and flag for alphabet normalisation. 
    Scores Euclidean distance between 
    sum of vectors for each symbol within the given string. 
    Afterwards multiplied by division of smaller entropy
    by bigger entropy, if alphabet normalisation flag
    is set to True. Splits by length of 
    the longer sequence. Finally, multiplies the result by 
    (1 - [Jaro distance between two sequences]), in order to 
    compensate for the lack of sequential information in
    vectors.

    Arguments:
        strings_info(StringSimilarityMeasurementInfo):
        an object with required parameters: str1, str2,
        vec1, vec2, ent1, ent2; and optional patameter
        alphabet_normalisation. For details, see 
        StringSimilarityMeasurementInfo description
    Returns:
        distance(float): distance between given sequences with
        vectors, alphabet entropy and Jaro distance
        between the sequences taken into consideration
    """
    if (not strings_info.str1 or not strings_info.str2):
        raise ValueError("One of strings is absent")
    if (not strings_info.vec1 or not strings_info.vec2):
        raise ValueError("One of vector dictionaries is absent")
    if ((not strings_info.ent1 or not strings_info.ent1) \
        and strings_info.alphabet_normalisation):
        raise ValueError("There is no alphabet entropy information")
    jaro = jw(strings_info.str1, strings_info.str2, winkler = False)/\
    max([len(strings_info.str1), len(strings_info.str2)])
    sum_s1 = sum(
    list(strings_info.vec1[v] for _, v in list(
    enumerate(strings_info.str1
                ))))
    sum_s2 = sum(
    list(strings_info.vec2[v] for _, v in list(
    enumerate(strings_info.str2
                ))))
    diff = euclidean(sum_s1, sum_s2)
    alphabet_difference = \
    strings_info.ent1/strings_info.ent2 \
        if strings_info.ent1 > strings_info.ent2 \
            else strings_info.ent2/strings_info.ent1
    length_normalisation = max([len(strings_info.str1), len(strings_info.str2)])
    final_vector_diff = diff/length_normalisation \
        if not strings_info.alphabet_normalisation \
            else (diff * alphabet_difference)/length_normalisation
    return final_vector_diff * (1 - jaro)
