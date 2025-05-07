"""
Vectorisation module aims at building vectors
for symbols of lects within a presented dataset,
in order to be utilised
in the further distance measurement
"""

from logging import getLogger, NullHandler
from os import mkdir
from os.path import exists, join
from re import sub
from math import log
from dataclasses import dataclass
from copy import deepcopy

from numpy import save as save_embeddings
from pandas import DataFrame
from gensim.models import FastText
from tqdm import tqdm

from corpus_distance.cdutils import get_lects_from_dataframe

logger = getLogger(__name__)
logger.addHandler(NullHandler())


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
    workers: int = 1
    epochs: int = 300
    seed: int = 42
    sg: int = 1

@dataclass
class Lect:
    """
    A model for representing key properties of the lect,
    required for the distance measurement
    * name (str): the name of the lect
    * alphabet (list[str]): a list of symbols in the writing system of the lect,
    joined by the additional CLS ^ and EOS $ symbols
    * alphabetic_vectors (dict): a FastText character-based embedding
    for each given symbol of the lect texts alphabet
    * alphabetic_entropy(int): amount of entropy of the lect texts alphabet
    """
    name: str
    alphabet: list[str]
    alphabetic_vectors: dict
    alphabet_entropy: int

def get_alphabet_entropy(text: str, alphabet: list[str]) -> int:
    """
    Scores the entropy of each symbol by
    - FREQ * log2(FREQ) formula
    and then sums entropy for all the
    symbols within a given lect

    Arguments:
        text (str): joined presented texts for the given lect
        alphabet (list[str]): a list of symbols in the writing system of the lect,
        joined by the additional CLS ^ and EOS $ symbols

    Returns:
        sum (int): an entropy measure
        of the lect texts alphabet provided
    """
    # add CLS and EOS symbols
    words = ['^' + word + '$' for word in text.split(' ')]
    # split everything by letter
    text_as_letters = [letter for word in words for letter in word]
    text_size = len(text_as_letters)
    entropy = []
    for symbol in alphabet:
        # score frequency for each given letter
        freq = len([letter for letter in text_as_letters if letter == symbol])/text_size
        # score its entropy by - FREQ * log2(FREQ) formula
        entropy.append(-freq * log(freq, 2))
    # sum entropy for all the symbols to get alphabet entropy
    return sum(entropy)

def get_letter_vectors_for_lect(text: str, alphabet: list[str],
        fasttext_params: FastTextParams = FastTextParams()) -> dict:
    """
    Produces chatacter-based embeddings for each symbol of the lect texts alphabet with
    FastText model, regulated with hyperparameters in the FastTextParams object.

    Arguments:
        text (str): joined presented texts for the given lect
        alphabet (list[str]): a list of symbols in the writing system of the lect,
        joined by the additional CLS ^ and EOS $ symbols
        fasttext_params (FastTextParams): a set of hyperparameters
        for FastText model: vector_size, window, min_count, 
        workers, epochs, seed, and sg
    Returns:
        by_symbol_dictionary(dict): a dictionary with symbols
        of lect alphabet as keys and their vectors as values
    """
    # split text into graphic words
    words = ['^' + word + '$' for word in text.split(' ')]
    # present each word as kind of sentence, with letters as tokens
    tokenised_words = [list(word) for word in words]
    # train FastText embeddings, as they are static and free of previous
    # influences, which is not true for Transformers
    model = FastText(tokenised_words,
                        vector_size=fasttext_params.vector_size,
                        window=fasttext_params.window,
                        min_count=fasttext_params.min_count,
                        workers=fasttext_params.workers,
                        epochs=fasttext_params.epochs,
                        seed=fasttext_params.seed,
                        sg=fasttext_params.sg)
    # create a dictionary of vectors for each symbol
    by_symbol_dictionary = {}
    for symbol in alphabet:
        by_symbol_dictionary[symbol] = model.wv[symbol]
    return by_symbol_dictionary


def save_vector_info_about_lect(output_dir: str, lect: Lect) -> None:
    """
    Stores the character-based embeddings and alphabet entropy in the experiment directory.
    It dumps character-based embeddings into a series of .npy files in a vectors subdirectory.
    The name for each .npy file is a hexadecimal representation of a symbol order number
    in Unicode. For example, file `0x46b.npy` is going to have embeddings for big jus (ѫ).
    This is due to the fact that some symbols (like slash \\) are not allowed in file names.  
    and alphabet entropy - to a common for all of the lects .tsv file
    (lect names are in the first column, denoted as Lect,
    corresponding entropy values are in the second column, denoted as Entropy).
    
    Arguments:
        output_dir (str): initial path to directory, where a package will store 
        the character-based embeddings for the symbols from the texts on the lect
        as a dictionary with character as key and 
        embeddings as value, and alphabet entropy value for the symbols in texts of the lect
        lect (Lect): An object of class Lect that contains the name of the analysed lect, 
        the character-based embeddings for the lect, and the alphabet entropy of the lect texts
    """
    if not exists(output_dir):
        raise ValueError("Storage directory does not exist")
    vector_dir_path = join(output_dir, lect.name + '_vectors')
    logger.debug("Creating directory to store the embeddings: %s", vector_dir_path)
    mkdir(vector_dir_path)
    for k, v in lect.alphabetic_vectors.items():
        symbol_in_unicode_encoding = str(hex(ord(k)))
        embeddings_path = join(
            vector_dir_path, f"{symbol_in_unicode_encoding}_embeddings.npy"
            )
        save_embeddings(embeddings_path, v)
        logger.debug(
            "Vectors for %s stored in %s",
            symbol_in_unicode_encoding,
            embeddings_path
            )
    entropy_path = join(output_dir, 'lect_entropies.tsv')
    if not exists(entropy_path):
        logger.debug(".tsv file for lect entropies %s does not exist, creating...", entropy_path)
        with open(entropy_path, "w", encoding="utf-8") as out:
            out.write("Lect\tEntropy\n")
    with open (entropy_path, "a", encoding="utf-8") as out:
        out.write(f"{lect.name}\t{str(lect.alphabet_entropy)}\n")
        logger.debug(
            "Alphabetic entropy for %s (%s) stored in %s",
            lect.name, lect.alphabet_entropy, entropy_path
        )

def create_vectors_for_given_lect(
        df: DataFrame, output_dir: str, lect_name: str,
        fasttext_params: FastTextParams = FastTextParams()
        ) -> Lect:
    """
    Creates FastText character-based embeddings for the lect, employing user-defined parameters
    for FastText model;
    it also adds information on alphabet entropy in the provided texts of the lect

    Arguments:
        df (DataFrame): a pandas DataFrame object that contains the texts, labeled by the lect
        output_dir (str): initial path to directory, where a package will store
        the lect character-based embeddings as a dictionary with character as key and 
        embeddings as value, and alphabet entropy value for the lect texts
        lect_name (str): the name of the lect, for which the FastText adds the 
        information on alphabet entropy and character-based embeddings
        fasttext_params (FastTextParams): the user-defined parameters for FastText model, which
        generates the character-based embeddings for the symbols of the lect

    Returns:
        An object of class Lect that contains the name of the analysed lect, 
        the character-based embeddings for the lect, and the alphabet entropy of the lect texts

    """
    lect_texts = []
    for _, row in df.iterrows():
        if lect_name == row['lect']:
            lect_texts.append(row['text'])
    lect_full_text = ' '.join(lect_texts)
    lect_lowered_full_text = lect_full_text.lower()
    # initialise list of lect symbols with special CLS ^ and EOS $ symbols
    lect_alphabet = ['^', '$']
    # add all the symbols from the texts
    lect_alphabet.extend(list(set(sub(' ', '', lect_lowered_full_text))))
    lect_alphabet_vectors = get_letter_vectors_for_lect(
        lect_lowered_full_text, lect_alphabet, fasttext_params
        )
    lect_alphabetic_entropy = get_alphabet_entropy(lect_lowered_full_text, lect_alphabet)
    lect_information = Lect(
        lect_name, lect_alphabet,
        lect_alphabet_vectors, lect_alphabetic_entropy
        )
    save_vector_info_about_lect(output_dir, lect_information)
    return lect_information

def create_vectors_for_lects(df: DataFrame, output_dir: str,
                            fasttext_params: FastTextParams = FastTextParams()
                            ) -> list[Lect]:
    """
    Creates a list of dictionaries, where each dictionary
    represents a lect in a given dataset.

    Arguments:
        df (DataFrame): a dataset with texts,
        matched with lects
        output_dir (str): initial path to directory, where a package will store
        the lect character-based embeddings as a dictionary with character as key and 
        embeddings as value, and alphabet entropy value for the lect texts
        fasttext_params (FastTextParams): the user-defined parameters for FastText model, which
        generates the character-based embeddings for the symbols of the lect

    Returns:
        lects_with_vectors (list[Lect]): 
        a list of dictionaries, each representing
        a corresponding lect with vectors and
        alphabet entropy
    """
    if 'lect' not in df.columns or 'text' not in df.columns:
        raise ValueError("No either \'lect\' or \'text\' columns")
    lects_names = get_lects_from_dataframe(df)
    lects_with_vectors = {}
    for lect in tqdm(lects_names):
        lects_with_vectors[lect] = create_vectors_for_given_lect(
            df, output_dir, lect, fasttext_params
            )
    return lects_with_vectors

def gather_vector_information(
        df: DataFrame,
        lects_with_vectors: list[Lect]) -> DataFrame:
    """
    Enriches the dataframe with information on symbolic vectors

    Arguments:
        df (DataFrame): original dataset
        lects_with_vectors (list[Lect]): 
        a list of dictionaries, each representing
        a corresponding lect with vectors and
        alphabet entropy
    Returns:
        vector_df (DataFrame): dataset enriched with 
        vectors for each symbol and alphabet
        entropy
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
     if l == x['lect']][0].alphabet_entropy,
                                    axis = 1)
    return vector_df
