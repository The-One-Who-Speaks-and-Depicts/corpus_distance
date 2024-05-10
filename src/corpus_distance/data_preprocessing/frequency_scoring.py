"""
Frequency scoring module enriches the dataset
with the information on n-gram frequency ranks
in the interval from 0 to 1
"""

from copy import deepcopy
from collections import Counter
from pandas import DataFrame
from corpus_distance import cdutils

def count_n_grams(array_of_n_grams: list) -> list[tuple]:
    """
    Scores frequency ranks for each n-gram in array

    Arguments:
        array_of_n_grams(list): n-grams from a lect
    Returns:
        rearranage_n_grams(list): set of tuples, 
        where the first value is n-gram, and
        the second is its frequency
    """
    # scoring number of appearances for each n-gram
    arranged_n_grams = Counter(array_of_n_grams).most_common()
    rearranged_n_grams = []
    # assigning rank based on frequency: the higher frequency, the closer to the head:
    # the most frequent n-gram gets rank 0, the second most frequent gets rank 0,
    # and so forth
    for i in list(enumerate(arranged_n_grams)):
        rearranged_n_grams.append((i[1][0], i[0]))
    return rearranged_n_grams

def count_n_grams_frequencies(df: DataFrame) -> DataFrame:
    """
    For each n-gram of each lect, returns its frequency
    in interval from 0 to 1

    Arguments:
        df(DataFrame): dataframe with n-grams column
        for each lect
    Returns:
        freq_df(DataFrame): dataframe with relative
        frequencies of n-grams for each lect
    """
    if 'n_grams' not in df.columns:
        raise ValueError("No \'n_gram\' column")
    freq_df = deepcopy(df)
    freq_df['frequency_arranged_n_grams'] = freq_df.apply(
        lambda x: count_n_grams(x['n_grams']), axis = 1
        )
    # in order for calculations to be more precise, ranks are transformed into
    # a [0;1] interval for each lect
    freq_df['relative_frequency_n_grams'] = freq_df.apply(
        lambda x: list(zip(
            [i[0] for i in x['frequency_arranged_n_grams']],
            cdutils.get_to_0_1([i[1] for i in x['frequency_arranged_n_grams']]),
        )),
        axis = 1
    )
    return freq_df
