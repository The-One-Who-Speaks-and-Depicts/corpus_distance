"""
Cdutils module contains functions that are used across 
the whole package and should be accessible from 
each part of the code; collected here to avoid
duplicating.
"""

import logging
from pandas import DataFrame
from numpy import percentile

def clear_stop_words(text: str, stop_words: list[str]) -> str:
    """
    Takes the text and returns it without given list of stopwords

    Arguments:
    text(str): an original text as a single string
    stop_words(list[str]): a list of strings, each being a stopword

    Returns:
    text(str): a text, cleared from stopwords
    """
    return ' '.join([i for i in text.split(' ') if i not in stop_words])

def get_lects_from_dataframe(df: DataFrame) -> list[str]:
    """
    Takes the dataframe with column "lect" 
    and returns the unique values of this column

    Arguments:
        df(DataFrame): a dataframe with a required column "lect"
    Returns:
        lects(list[str]): a list with names of lects
    """
    if 'lect' not in df.columns:
        raise ValueError("No column named \'lect\' in dataframe")
    return list(df['lect'].unique())

def get_to_0_1(distribution: list[int|float]) -> list [int|float] :
    """
    A function that helps to rescale the values to [0;1]. I use 
    special kind of rescaling to get more normal distribution, not the one
    when the smallest value in original distribution is 0, 
    and the biggest is 1

    Arguments:
        distribution (list[int|float]): original list of 
        numeric values

    Returns:
        rescaled_distribution(list[int|float]): list of numeric values,
        rescaled to [0;1] space  
    """
    q1, q3 = percentile(distribution, [25,75])
    iqr=q3-q1
    minimum=min(distribution)
    maximum=max(distribution)
    return [((i - minimum + iqr)/(maximum - minimum + 2*iqr)) for i in distribution]



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
    if len(original_distribution) < 1:
        raise ValueError("There is no elements in the original list")
    q1, q3 = percentile(original_distribution, [25,75])
    interquartile_range = q3 - q1
    upper_boundary = q3 + 1.5 * interquartile_range
    lower_boundary = q1 - 1.5 * interquartile_range
    normalised_distribution =\
        [
            i for i in original_distribution
            if not lower_boundary <= i <= upper_boundary
        ]
    if len(normalised_distribution) < 1:
        logging.warning("Original distribution is not a normal distribution")
        return original_distribution
    return normalised_distribution


def get_unique_pairs(lects: list[str]) -> list[tuple[str, str]]:
    """
    Acquires pairs of lects from given list to compare

    Parameters:
        lects(list[str]): a list of lect names
    Returns:
        unique_pairs(list[tuple[str, str]]): a list of 
        lect pairs, used for analysis
    """
    if not lects:
        raise ValueError("Empty lect list")
    unique_pairs = []
    for i in set(lects):
        for j in set(lects):
            if (i != j) and (i, j) not in unique_pairs and (j, i) not in unique_pairs:
                unique_pairs.append((i, j))
    logging.info("Unique pairs: %s", ";".join([i[0] + i[1] for i in unique_pairs]))
    return unique_pairs
