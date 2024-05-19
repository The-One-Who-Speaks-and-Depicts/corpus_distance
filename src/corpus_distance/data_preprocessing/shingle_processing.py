"""
Shingle processing module aims at splitting the text by n-grams
(character 3-grams) for the purposes of enhancing dataset size and 
presence of variation variables within it. Also adds frequency-based
information to the given lect
"""

from pandas import DataFrame

from corpus_distance.cdutils import get_lects_from_dataframe

def n_gram_split(text: str) -> list[str]:
    """
    The first stage of data preprocessing is splitting tokens into character 3-grams. 
    The character n-grams help to find coinciding sequences more easily, 
    than tokens or token n-grams. 
    Specifically 3-grams help to underscore the exact places where the change is happening, 
    providing minimal left and right context for each symbol within the sequence. 
    Adding special symbols ^ and $ to the start and the end of each sequence 
    helps to do this for
    the first and the last symbol of the given sequence as well.

    Arguments:
        text(str): preliminarily tokenised text, where each token
        is split by space
    Returns:
        n_grams(list[str]): list of n-grams from a given text
    """
    n_grams = []
    for j in text.split():
        if j and j.strip():
            s = list(j)
            # deleting void first symbol, if present
            if ord(j[0]) == 65279:
                s.pop(0)
            s = ''.join(s)
            # assigning 3-gram for each symbol within the sequence
            for k in list(enumerate(s)):
                # if the sequence consists of only one symbol,
                # surrounding it by special tokens
                if k[0] == 0 and (len(s) == 1):
                    n_grams.append(''.join(['^', k[1],'$']))
                    continue
                # if the current symbol is the first within the sequence,
                # add special token ^ before it
                if k[0] == 0:
                    n_grams.append(''.join(['^', k[1], s[k[0] + 1]]))
                    continue
                # if the current symbol is the last within the sequence,
                # add special token $ after it
                if k[0] == (len(s) - 1):
                    n_grams.append(''.join([s[k[0] - 1], k[1], '$']))
                    continue
                # in any other case, return
                # previous, current and following symbols
                n_grams.append(''.join([s[k[0] - 1], k[1], s[k[0] + 1]]))
    return n_grams

def assign_n_grams_to_lects(df: DataFrame) -> dict:
    """
    Provides an n-grams array for each given lect

    Arguments:
        df(DataFrame): a dataframe with lect and n-grams
        columns

    Returns:
        n_grams_by_lects(dict) : a dictionary with
        lect names as keys and n-gram arrays 
        as values
    """
    if 'lect' not in df.columns or 'n_grams' not in df.columns:
        raise ValueError("No either \'lect\' or \'n_grams\' columns")
    lects = get_lects_from_dataframe(df)
    n_grams_by_lects = {}
    for l in lects:
        joined_n_grams = list(df[df['lect'] == l]['n_grams'])
        n_grams_for_lect = [j for i in joined_n_grams for j in i]
        n_grams_by_lects[l] = n_grams_for_lect
    return n_grams_by_lects


def split_lects_by_n_grams(df: DataFrame) -> DataFrame:
    """
    Takes a dataframe of text/lect correspondences,
    and transforms it into a dataframe of 
    lect/n-gram correspondences
    
    Arguments:
        df(DataFrame): an original dataframe with text/lect
        correspondences
    Returns:
        n_gram_df(DataFrame): a transformed dataframe
        with lect/n-gram correspondences
    """
    if 'lect' not in df.columns or 'text' not in df.columns:
        raise ValueError("No either \'lect\' or \'text\' columns")
    df['n_grams'] = df.apply(lambda x: n_gram_split(x['text']), axis=1)
    n_grams_by_lects = assign_n_grams_to_lects(df)
    return DataFrame(n_grams_by_lects.items(), columns=['lect', 'n_grams'])
