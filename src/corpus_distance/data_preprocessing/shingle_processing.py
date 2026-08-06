"""
Shingle processing module aims at splitting the text by character n-grams
for the purposes of enhancing dataset size and 
presence of variation variables within it.

The module contains the following functions:

* ENTRY: split_lects_by_n_grams(df: pd.DataFrame, n: int = 3) -> pd.DataFrame: splits the texts
in dataframe into character sequences (default n = 3).
* n_gram_split(text: str, n: int = 3) -> list[str]: splits a given text string
into a sequence of character n-grams, with default n being 3
* preprocess_token_for_split(token: list[str]) -> list[str]: deletes problematic technical
symbols and adds CLS (^) and EOS ($) symbols
* assign_n_grams_to_lects(df: pd.DataFrame) -> dict: extracts information
on lects and their respective n-grams from given pandas DataFrame (with columns
n_grams and lect)

"""
from logging import getLogger, NullHandler

from numpy import array, issubdtype, integer, ndarray
from pandas import DataFrame

from corpus_distance.cdutils import get_lects_from_dataframe

logger = getLogger(__name__)
logger.addHandler(NullHandler())


def preprocess_token_for_split(token: list[str]) -> list[str]:
    """
    Takes the list of symbols, deletes void symbols that randomly appear
    in the beginning of the string,
    and adds ^ and $ symbols as CLS and EOS.

    Arguments:
        token(list[str]): string as a sequence of characters.

    Returns:
        list[str]: sequence of characters with CLS and EOS, but without void symbol.
    """
    logger.debug("Input params: %s", locals())
    # first of all, it is necessary to delete a special unicode empty symbol
    if ord(token[0]) == 65279:
        token.pop(0)
    # wrap token with special symbols of beginning and end
    token.insert(0, "^")
    token.append("$")
    logger.debug("Result: %s", token)
    return token


def n_gram_split(text: str, n: int = 3) -> list[str]:
    """
    The first stage of data preprocessing is splitting tokens into character n-grams. 
    The character n-grams help to find coinciding sequences more easily, 
    than tokens or token n-grams. Adding special symbols ^ and $ to the start
    and the end of each sequence helps to do this for
    the first and the last symbol of the given sequence as well.

    Arguments:
        text(str): preliminarily tokenised text, where each token
        is split by space
    Returns:
        n_grams(list[str]): list of n-grams from a given text
    """
    logger.debug("Input params: %s", locals())
    if not isinstance(n, int):
        raise ValueError(f"n should be a positive integer, received {n}")
    if n < 1:
        raise ValueError(f"n should be a positive integer, received {n}")
    n_grams = []
    # the next string splits sequence into pre-detected tokens,
    # and transforms these tokens into a list of characters each
    # to form character n-grams
    split_tokens = [list(tok) for tok in text.split() if tok and tok.strip()]
    for token in split_tokens:
        processed_token = preprocess_token_for_split(token)
        # if there is more symbols left in token than n, required for n-grams,
        # the algorithm takes exactly n symbols and deletes the first one
        # from the token
        while len(processed_token) > n:
            n_grams.append(''.join(processed_token[0:n]))
            processed_token.pop(0)
        # otherwise, it just returns the rest
        n_grams.append(''.join(processed_token))
    logger.debug("Result: %s", token)
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
    logger.debug("Input params: %s", locals())
    if 'lect' not in df.columns or 'n_grams' not in df.columns:
        raise ValueError("No either \'lect\' or \'n_grams\' columns")
    lects = get_lects_from_dataframe(df)
    n_grams_by_lects = {}
    for l in lects:
        joined_n_grams = list(df[df['lect'] == l]['n_grams'])
        n_grams_for_lect = [j for i in joined_n_grams for j in i]
        n_grams_by_lects[l] = n_grams_for_lect
    logger.debug("Result: %s", n_grams_by_lects)
    return n_grams_by_lects


def split_lects_by_n_grams(df: DataFrame, n: int = 3) -> DataFrame:
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
    logger.debug("Input params: %s", locals())
    if 'lect' not in df.columns or 'text' not in df.columns:
        raise ValueError("No either \'lect\' or \'text\' columns")
    df['n_grams'] = df.apply(lambda x: n_gram_split(x['text'], n), axis=1)
    n_grams_by_lects = assign_n_grams_to_lects(df)
    result = DataFrame(n_grams_by_lects.items(), columns=['lect', 'n_grams'])
    logger.debug("Result: %s", n_grams_by_lects)
    return result

def get_length_of_shingle_list(data_frame: DataFrame, lect: str) -> int:
    logger.debug("Input params: %s", locals())
    if not isinstance(
        data_frame, DataFrame
        ) or not 'lect' in data_frame.columns or not 'n_grams' in data_frame.columns:
        raise ValueError("data_frame should be a pandas DataFrame " \
        f"with columns \'lect\' and \'n_grams\', received{data_frame}")
    if not isinstance(lect, str) or not lect.strip() or not lect in list(data_frame["lect"]):
        raise ValueError("lect should be a non-empty string present in the column \'lect\' "
        f"of data_frame, received {lect}")
    for _, row in data_frame.iterrows():
        if row['lect'] == lect:
            result = len(list(set(row['n_grams'])))
            logger.debug("Result: %s", result)
            return result

def get_n_shingles_number_by_lect(
        data_frame: DataFrame, init: int, limit: int
        ) -> dict[str, ndarray]:
    logger.debug("Input params: %s", locals())
    if not isinstance(
            data_frame, DataFrame
            ) or not 'lect' in data_frame.columns or not 'text' in data_frame.columns:
            raise ValueError("data_frame should be a pandas DataFrame " \
            f"with columns \'lect\' and \'text\', received{data_frame}")
    if not issubdtype(type(init), integer) or init < 1:
        raise ValueError(f"init should be a positive integer, received {init}")
    if not issubdtype(type(limit), integer) or limit <= init:
            raise ValueError(f"limit should be a positive integer bigger than init({init}), " \
            f"received {limit}")
    lects = get_lects_from_dataframe(data_frame)
    quantities_dict = {lect: [] for lect in lects}
    for idx in range(init, limit):
        lects_split_by_n_grams = split_lects_by_n_grams(data_frame, idx)
        for lect in lects:
            length_n_grams_lect = get_length_of_shingle_list(lects_split_by_n_grams, lect)
            quantities_dict[lect].append(length_n_grams_lect)
    logger.debug("Result: %s", quantities_dict)
    return quantities_dict