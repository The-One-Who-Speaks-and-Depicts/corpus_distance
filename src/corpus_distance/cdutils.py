"""
Cdutils module contains functions that are used across
the whole package and should be accessible from
each part of the code; collected here to avoid
duplicating.

Functions:
    clear_stop_words(text: str, stop_words: list[str] = None) -> str:
    takes a string and a sequence of strings, and clears the strings
    from this sequence from the string. In case the sequence is empty,
    returns the original string.
    return_topic_words(text: str, topic_words: list[str] = None) -> str:
    takes a string and a sequence of strings, and retains only the strings
    from this sequence in the original string. In case the sequence is empty,
    raises ValueError.
    get_lects_from_dataframe(data_frame: pandas.DataFrame) -> list[str]:
    takes a dataframe with column 'lect' in it, and returns unique values
    of this column.
    get_to_0_1(distribution: list[int|float]) -> list [int|float]: takes a distribution
    and scales it to [0;1] interval for the compatibility.
    delete_outliers (list[int|float]) -> list[int|float]: takes a distribution,
    and, if it is normal (Gaussian), deletes outliers from it, in case such are present in the data.
    Otherwise, returns the original distribution.
    get_unique_pairs(list[str]) -> list[LectPair]: takes a list of objects designated as lects,
    and transforms it into a list of non-repeating pairs.
    create_and_set_storage_directory(str) -> str: creates a directory to store the results
    of the experiments. In case the directory already exists, raises the warning. Returns
    a path to the directory, in case a user needs it.
"""
from copy import deepcopy
from os import mkdir, listdir
from os.path import exists
from warnings import warn
from logging import getLogger, NullHandler
from typing import NamedTuple

from numpy import issubdtype, number
from pandas import DataFrame
from numpy import percentile
from tqdm import tqdm

logger = getLogger(__name__)
logger.addHandler(NullHandler())

def clear_stop_words(text: str, stop_words: list[str] = None) -> str:
    """
    Takes a text and returns it without a given list of stopwords.

    Arguments:
    text(str): an original text as a single string
    stop_words(list[str]): a list of strings, each being a stopword

    Returns:
    text(str): a text, cleared from stopwords
    """
    logger.debug("Input params: %s", locals())
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"text should be a non-empty string, received {text}")
    if stop_words is None or (isinstance(stop_words, list) and len(stop_words) < 1):
        logger.warning("stop_words are absent from the input")
        stop_words = []
    if not isinstance(stop_words, list):
        raise ValueError(f"stop_words parameter is not a list, received {stop_words}")
    result = ' '.join([i for i in text.split(' ') if i not in stop_words])
    logger.debug("Result: %s", result)
    return result


def return_topic_words(text: str, topic_words: list[str]) -> str:
    """
    Takes the text and returns only topic words from it,
    separated by a single space

    Arguments:
    text(str): an original text as a single string
    topic_words(list[str]): a list of strings, each being a topic word

    Returns:
    text(str): a text, containing only topic words
    """
    logger.debug("Input params: %s", locals())
    if not isinstance(text, str) or (
        isinstance(text, str) and not text.strip()
    ):
        raise ValueError(f"text should be a non-empty string, received {text}")
    if not isinstance(topic_words, list) or (
        isinstance(topic_words, list) and len(topic_words) < 1
        ):
        raise ValueError(f"topic_words parameter is not a non-empty list, received {topic_words}")
    result = ' '.join([i for i in text.split(' ') if i in topic_words])
    logger.debug("Result: %s", result)
    return result


def get_lects_from_dataframe(data_frame: DataFrame) -> list[str]:
    """
    Takes the dataframe with column "lect"
    and returns the unique values of this column

    Arguments:
        df(DataFrame): a dataframe with a required column "lect"
    Returns:
        lects(list[str]): a list with names of lects
    """
    logger.debug("Input params: %s", locals())
    if not isinstance(data_frame, DataFrame):
        raise ValueError("Argument data_frame should be a pandas data frame,"\
                         f"received {type(data_frame)}")
    if 'lect' not in data_frame.columns:
        raise ValueError("No column named \'lect\' in the data frame,"\
                         f"received {data_frame.columns}")
    result = list(data_frame['lect'].unique())
    logger.debug("Result: %s", result)
    return result

def get_to_0_1(distribution: list[int|float]) -> list [int|float]:
    """
    A function that helps to rescale the values to [0;1]. I use
    special kind of rescaling to get more normal distribution, not the one
    when the smallest value in original distribution is 0,
    and the biggest is 1.

    Arguments:
        distribution (list[int|float]): original list of
        numeric values

    Returns:
        rescaled_distribution(list[int|float]): list of numeric values,
        rescaled to [0;1] space
    """
    logger.debug("Input params: %s", locals())
    if not isinstance(distribution, list) or not all(
        issubdtype(type(x), number) for x in distribution
        ):
        raise ValueError(f"Distribution should be a list of integers, got {distribution} instead")
    # There is no need to perform normalisation per se, so if the distribution
    # is in [0; 1] rank, I leave it as such.
    if all(0 <= x <= 1 for x in distribution):
        return distribution
    # The most problematic case is a distribution with zero variance,
    # which can occur sometimes (with small data, much more often than
    # not). There is no good solution to that (and, hopefully, no
    # need to implement it in the particular package, with DistRanks
    # normalised beforehand), so the one I am implementing is a rather
    # crude one: if the values are already in [0; 1], the previous line
    # got it covered. All the other values become 1. I still
    # raise warning, because the user is going to need this information.
    if all(x == distribution[0] for x in distribution):
        logger.warning(
            "All the values in the original distribution equal to %s,"\
            "scaling the distribution to a single value",
            distribution[0]
            )
        return [1 for _ in range(len(distribution))]
    first_quartile, third_quartile = percentile(distribution, [25,75])
    interquartile_difference = third_quartile - first_quartile
    minimum=min(distribution)
    maximum=max(distribution)
    result = [(
        (
            i - minimum + interquartile_difference
            )/(
                maximum - minimum + 2*interquartile_difference
                )
                ) for i in distribution]
    logger.debug("Result: %s", result)
    return result



def delete_outliers(original_distribution: list[int|float]
                          ) -> list[int|float]:
    """
    An auxiliary function that deletes outliers from
    the list of metrics values.

    Arguments:
        original_distribution (list[int|float]): a list of
        metric values that satisfies normal distribution criteria,
        but has some outliers
    Returns:
        normalised_distribution(list[int|float]): an original list,
        cleared from the outliers
    """
    logger.debug("Input params: %s", locals())
    if not isinstance(original_distribution, list):
        raise ValueError(
            f"The input type should be a list, got {original_distribution}"
            )
    if any(not issubdtype(type(x), number) for x in original_distribution):
        raise ValueError(
            f"The input type should be a numerical list, got {original_distribution}"
        )
    if len(original_distribution) < 1:
        raise ValueError("There is no elements in the original list")
    first_quartile, third_quartile = percentile(original_distribution, [25,75])
    interquartile_range = third_quartile - first_quartile
    upper_boundary = third_quartile + 1.5 * interquartile_range
    lower_boundary = first_quartile - 1.5 * interquartile_range
    normalised_distribution =\
        [
            i for i in original_distribution
            if lower_boundary < i < upper_boundary
        ]
    if len(normalised_distribution) < 1:
        logger.warning("Original distribution is not a normal distribution")
        return original_distribution
    logger.debug("Result: %s", normalised_distribution)
    return normalised_distribution

class LectPair(NamedTuple):
    """
    Designed to store the names of two compared lects in string form.
    """
    lect1: str
    lect2: str

def get_unique_pairs(lects: list[str]) -> list[LectPair]:
    """
    Acquires pairs of lects from given list to compare.

    Parameters:
        lects(list[str]): a list of lect names.
    Returns:
        unique_pairs(list[LectPair]): a list of
        lect pairs, used for analysis.
    """
    logger.debug("Input params: %s", locals())
    if not isinstance(lects, list) or any(not isinstance(i, str) for i in lects):
        raise ValueError(f"Lects should be a list of strings, got {lects}")
    if len(lects) < 2:
        raise ValueError("Not enough lects to form pairs")
    unique_pairs = []
    lects_to_check = deepcopy(lects)
    for i in tqdm(lects):
        for j in lects_to_check:
            if i != j:
                unique_pairs.append((i, j))
        lects_to_check = [k for k in lects_to_check if k != i]
    logger.info("Unique pairs: %s", ";".join([i[0] + i[1] for i in unique_pairs]))
    return unique_pairs


def create_and_set_storage_directory(store_path: str) -> str:
    """
    Sets directory for experiment results, in case of its absence,
    creates it. In case the directory is not empty, throws a warning,
    but stores files in the directory nonetheless.

    Parameters:
        store_path(str): initial path to directory, where a package will store the results
    Returns:
        store_path(str): final path to directory, where a package will store the results
    """
    if not isinstance(store_path, str) or not store_path.strip():
        raise ValueError("Storage directory name is not a non-empty string")
    if not exists(store_path):
        logger.info("Creating directory %s", store_path)
        mkdir(store_path)
    if len(listdir(store_path)) > 0:
        warn(
            f"Storage directory {store_path} is not empty, consider choosing the other one"
            )
    return store_path
