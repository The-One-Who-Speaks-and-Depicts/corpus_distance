"""
Data loading model contains functions load_data
and load_default_data.
Load_data is used for loading user-defined data in a specific format,
while load_default_data performs the same transformations for
a demo dataset of three standard Slavic Gospels (Slovak, Slovenian, Croatian) 
"""



from os import listdir
from os.path import join, isfile, isdir
from logging import getLogger, NullHandler
from itertools import islice
from math import ceil
from pandas import DataFrame
import corpus_distance.data.data_resources as datares

logger = getLogger(__name__)
logger.addHandler(NullHandler())

def load_data(content_directory: str , split: int | float = 1) -> DataFrame:
    """
    Takes directory and size share,
    and returns a dataframe with texts 
    as first column and lect names as second. 
 
    Args:
        content_directory (string): path to the directory with files of the lects.
        Files should have TEXT.LECT.txt style of naming.
        For example, Gospel.Croatian.txt.

        split (int): share (from 0 (not inclusive) to 1 (inclusive)).
 
    Returns:
        df: A dataframe with texts as first column and lect names as second. 
    """
    logger.debug("Input parameters: %s", locals())
    if not isinstance(content_directory, str) or not content_directory.strip():
        raise ValueError(
            f"content_directory should be a non-empty string, received {content_directory}"
            )
    if not isdir(content_directory):
        raise ValueError("content_directory is not an existing path to a folder!")
    # disable undidiomatic typecheck, because otherwise the code is going to become a boilerplate,
    # or I am going to get boolean go further
    # pylint: disable=unidiomatic-typecheck
    if not (type(split) is int or type(split) is float):
        raise ValueError(f"split should be an integer or a float, received {split}")
    if not 0 < split <= 1:
        raise ValueError("split argument has an incorrect value, the allowed range is from 0 to 1")
    texts = {}
    for filename in listdir(content_directory):
        f = join(content_directory, filename)
        # checking if it is a txt file
        if isfile(f) and f.endswith('.txt'):
            logger.info("Preprocessing file %s", f)
            split_file_name = f.split('.')
            if len(split_file_name) < 3:
                raise ValueError(f"File {f} should have been named "\
                                 "according to the TEXT.LECT.txt template"\
                                 f", received {f}")
            lect = filename.split('.')[-2]
            with open(f, 'r', encoding='utf-8') as inp:
                content = inp.read().lower().strip().split(' ')
                split_text = ' '.join(list(islice(content, ceil(len(content)*split))))
                texts[split_text] = lect
    df = DataFrame(texts.items(), columns=['text', 'lect'])
    # If there is no data, proceeding further with the pipeline may cause a user's confusion.
    # However, in case the actual implementation loads default data, if no data is available,
    # throwing an error would become too enforcing. Therefore, I put the logger warning.
    if df.shape[0] == 0:
        logger.warning("No data loaded, proceed with caution")
    logger.debug("Data loaded: %s", df)
    return df


def load_default_data() -> DataFrame:
    """

    Wrapping of the default data for simplified use.
    Default data is the Croatian, Slovak and Slovenian 
    John's Gospels, each containing
    approximately 19,000 tokens.

    Returns: 
        df: A dataframe with texts as first column and lect names as a second. 

    """
    result = datares.data_df
    logger.debug("Default data loaded: %s", result)
    return result
