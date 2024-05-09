import os
from itertools import islice
from math import ceil
from pandas import DataFrame

def load_data(content_directory: str , split: int = 1) -> DataFrame:
    """
    Takes directory and size share,
    and returns a dataframe with texts 
    as first column and lect names as second. 
 
    Args:
        content_directory (string): path to the directory with files of the lects.
        Files should have TEXT.LECT.txt style of namingю
        For example, Gospel.Croatian.txt.

        split (int): share (from 0 to 1).
 
    Returns:
        df: A dataframe with texts as first column and lect names as second. 
    """
    if (split < 0 or split > 1):
        raise ValueError("Incorrect split, should be between 0 and 1")
    texts = {}
    for filename in os.listdir(content_directory):
        f = os.path.join(content_directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            lect = filename.split('.')[-2]
            with open(f, 'r', encoding='utf-8') as inp:
                content = inp.read().lower().strip().split(' ')
                split_text = ' '.join(list(islice(content, ceil(len(content)*split))))
                texts[split_text] = lect
    df = DataFrame(texts.items(), columns=['text', 'lect'])
    del texts
    return df

# TODO: import default data
