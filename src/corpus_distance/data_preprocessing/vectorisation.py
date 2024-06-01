"""
Vectorisation module aims at building vectors
for symbols of lects within a presented dataset,
in order to be utilised
in the further distance measurement
"""


from re import sub
from math import log
from dataclasses import dataclass
from copy import deepcopy

from pandas import DataFrame
from gensim.models import FastText
from tqdm import tqdm

from corpus_distance.cdutils import get_lects_from_dataframe




@dataclass
class FastTextParams:
    """
    Parameters for FastText model:
    * vector_size
    * window
    * min_count
    * workers
    * epochs
    * seed
    * sg
    For further details on each, refer to 
    FastText documentation
    """
    vector_size: int = 128
    window: int = 15
    min_count: int = 3
    workers: int = 4
    epochs: int = 300
    seed: int = 42
    sg: int = 1



class Lect:
    """
    A model for representing key properties of the lect
    * name
    * joined presented texts
    * alphabet (with additional CLS ^ and EOS $ symbols)
    * amount of enthropy of its alphabet
    * vector for each given symbol of the alphabet
    """

    def __init__(self, name: str, inp: str,
                fasttext_params: FastTextParams):
        """
        Arguments:
            name(str): name of the lect
            inp(str): texts of the lect,
            gathered to a single string
        """
        self.lect = name
        self.text = inp.lower()
        self.alphabet = ['^', '$']
        self.alphabet.extend(list(set(sub(' ', '', inp))))
        self.alphabet_information = self.get_alphabet_information()
        self.alphabetic_vectors = self.get_letter_vectors_for_lect(fasttext_params)

    def get_alphabet_information(self) -> int:
        """
        Scores the information of each symbol by
        - FREQ * log2(FREQ) formula
        and then sums information for all the
        symbols within a given lect

        Returns:
            sum(int) - an information measure
            of the lect provided
        """
        # add CLS and EOS symbols
        words = ['^' + i + '$' for i in self.text.split(' ')]
        # split everytig by letter
        text_as_letters = [i for word in words for i in word]
        text_size = len(text_as_letters)
        information = []
        for i in self.alphabet:
            # score frequency for each given letter
            freq = len([l for l in text_as_letters if l == i])/text_size
            # score its information by - FREQ * log2(FREQ) formula
            information.append(-freq * log(freq, 2))
        # sum information for all the symbols to get alphabet information
        return sum(information)

    def get_letter_vectors_for_lect(self,
            fasttext_params: FastTextParams = FastTextParams()) -> dict:
        """
        Produces vectorisation for each symbol of lect alphabet with
        FastText model, regulated by FastTextParams object, 
        that contains hyperparameters for FastText embeddings.

        Arguments:
            fasttext_params(FastTextParams): a set of hyperparameters
            for FastText model: vector_size, window, min_count, 
            workers, epochs, seed, and sg
        Returns:
            by_letter_dictionary(dict): a dictionary with symbols
            of lect alphabet as keys and their vectors as values
        """
        # split text into graphic words
        words = ['^' + i + '$' for i in self.text.split(' ')]
        # present each word as kind of sentence, with letters as tokens
        tokenised_words = [list(i) for i in words]
        # train FastText embeddings, as they are static and free of previous
        # influences, which is not true for BART
        model = FastText(tokenised_words,
                         vector_size=fasttext_params.vector_size,
                         window=fasttext_params.window,
                         min_count=fasttext_params.min_count,
                         workers=fasttext_params.workers,
                         epochs=fasttext_params.epochs,
                         seed=fasttext_params.seed,
                         sg=fasttext_params.sg)
        # create a dictionary of vectors for each letter
        by_letter_dictionary = {}
        for i in self.alphabet:
            by_letter_dictionary[i] = model.wv[i]
        return by_letter_dictionary


def create_vectors_for_lects(df: DataFrame,
                            fasttext_params: FastTextParams = FastTextParams()
                            ) -> list[Lect]:
    """
    Creates a list of dictionaries, where each dictionary
    represents a lect in a given dataset.

    Arguments:
        df(DataFrame): a dataset with texts,
        matched with lects
    Returns:
        lects_with_vectors(list[Lect]): 
        a list of dictionaries, each representing
        a corresponding lect with vectors and
        alphabet enthropy
    """
    if 'lect' not in df.columns or 'text' not in df.columns:
        raise ValueError("No either \'lect\' or \'text\' columns")
    lects_names = get_lects_from_dataframe(df)
    lects_with_vectors = {}
    for l in tqdm(lects_names):
        lect_texts = []
        for _, row in df.iterrows():
            if l == row['lect']:
                lect_texts.append(row['text'])
        lect_full_text = ' '.join(lect_texts)
        lects_with_vectors[l] = Lect(
            l, lect_full_text, fasttext_params)
    return lects_with_vectors

def gather_vector_information(
        df: DataFrame,
        lects_with_vectors: list[Lect]) -> DataFrame:
    """
    Enriches the dataframe with information on symbolic vectors

    Arguments:
        df(DataFrame): original dataset
        lects_with_vectors(list[Lect]): 
        a list of dictionaries, each representing
        a corresponding lect with vectors and
        alphabet enthropy
    Returns:
        vector_df(DataFrame): dataset enriched with 
        vectors for each symbol and alphabet
        enthropy
    """
    if 'lect' not in df.columns:
        raise ValueError("No \'lect\' column")
    vector_df = deepcopy(df)
    vector_df['lect_vectors'] = vector_df.apply(lambda x:
    [lects_with_vectors[l] for l in lects_with_vectors.keys()
     if l == x['lect']][0].alphabetic_vectors,
                                        axis = 1)
    vector_df['lect_info'] = vector_df.apply(lambda x:
    [lects_with_vectors[l] for l in lects_with_vectors.keys()
     if l == x['lect']][0].alphabet_information,
                                    axis = 1)
    return vector_df
