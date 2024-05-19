"""
Data pipeline module gathers all the data preprocessing functions
into a single straightforward transformation for
a more comfortable user experience
"""
from pandas import DataFrame
import corpus_distance.data_preprocessing.data_loading as loading
from corpus_distance import cdutils
import corpus_distance.data_preprocessing.shingle_processing as sp
import corpus_distance.data_preprocessing.vectorisation as vec
import corpus_distance.data_preprocessing.topic_modelling as tm
import corpus_distance.data_preprocessing.frequency_scoring as freqscore



def assemble_dataset(
        path_to_folder: str = 'default',
        split: int = 1,
        lda_params: tm.LDAParams = tm.LDAParams(),
        topic_modelling: bool = False,
        fasttext_params: vec.FastTextParams = vec.FastTextParams()) -> DataFrame:
    """
    Performs data processing for lects in given folder. If no folder is provided,
    uses default dataset.

    Arguments:
        path_to_folder(str): path to the folder with data
        lda_params(LDAParams): parameters for LDA model to learn on
        topic_modelling(bool): replacing of original text with 
        text, cleared from topic words
        fasttext_params(FastTextParams): parameters for FastText model
        to learn on
    Returns:
        df(DataFrame): a dataframe with information on n-gram frequencies,
        symbol vectors, and alphabet enthropy, optionally cleared from 
        the topic words
    """
    df = loading.load_default_data() \
        if path_to_folder == 'default' \
        else loading.load_data(path_to_folder, split)
    lects = cdutils.get_lects_from_dataframe(df)
    lects_with_topics = tm.get_topic_words_for_lects(df, lects, lda_params)
    df = tm.add_thematic_modelling(df, lects_with_topics, topic_modelling)
    vecs = vec.create_vectors_for_lects(df, fasttext_params)
    df = sp.split_lects_by_n_grams(df)
    df = freqscore.count_n_grams_frequencies(df)
    df = vec.gather_vector_information(df, vecs)
    return df
