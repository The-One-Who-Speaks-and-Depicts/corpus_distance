"""
Data pipeline module gathers all the data preprocessing functions
into a single straightforward transformation for
a more comfortable user experience
"""
from dataclasses import dataclass
from os.path import exists
from pandas import DataFrame
import corpus_distance.data_preprocessing.data_loading as loading
from corpus_distance import cdutils
import corpus_distance.data_preprocessing.shingle_processing as sp
import corpus_distance.data_preprocessing.vectorisation as vec
import corpus_distance.data_preprocessing.topic_modelling as tm
import corpus_distance.data_preprocessing.frequency_scoring as freqscore

@dataclass
class DatasetPreprocessingParams:
    """
    Class with parameters that define the transformations the dataset
    undergoes during the preprocessing stage.

    Parameters:
        content_path (str): path to directory with text files; text files should be
        named as "TEXT.LECT.txt", and consist of tokenised texts, transformed into
        a single string
        split (int|float): a number from 0 to 1, which signals, which percentage of
        source data should form the basis for clusterisation
        topic_modelling (string): flag that describes the choice of user 
        to change original text to text without topic words, not to change it,
        or use topic words only
    """
    store_path: str = "default"
    content_path: str = "default"
    split: int | float = 1
    topic_modelling: str = "not_substitute"

@dataclass
class DataParameters:
    """
    Class with parameters that define data loading and preprocessing
    part of pipeline

    Parameters:
        dataset_params (DatasetPreprocessingParams): a set of parameters for dataset preprocessing,
        for details see DatasetPreprocessingParams documentation
        lda_params (LDAParams): a set of parameters for latent dirichlet association
        model of gensim package, for details see LDAParams documentation
        fasttext_params (FastTextParams): a set of parameters for FastText model that
        builds symbol vectors, for details see FastText documentation
    """
    dataset_params: DatasetPreprocessingParams = DatasetPreprocessingParams()
    lda_params: tm.LDAParams = tm.LDAParams()
    fasttext_params: vec.FastTextParams = vec.FastTextParams()

def assemble_dataset(
        data_params: DataParameters = DataParameters()
        ) -> DataFrame:
    """
    Performs data processing for lects in given folder. If no folder is provided,
    uses default dataset.

    Arguments:
        data_params (DataParameters): a set of parameters that define
        the preprocessing of the dataset, as well as parameters for utilised
        machine learning models
    Returns:
        df (DataFrame): a dataframe with information on n-gram frequencies,
        symbol vectors, and alphabet entropy, optionally cleared from 
        the topic words
    """
    if not isinstance(data_params.dataset_params.store_path, str) \
        or not data_params.dataset_params.store_path \
        or not exists(data_params.dataset_params.store_path):
        raise ValueError("Path to experiment folder should exist")
    df = loading.load_default_data() \
        if data_params.dataset_params.content_path == 'default' \
        else loading.load_data(
            data_params.dataset_params.content_path,
            data_params.dataset_params.split
            )
    lects = cdutils.get_lects_from_dataframe(df)
    lects_with_topics = tm.get_topic_words_for_lects(
        df, lects, data_params.lda_params
        )
    df = tm.add_topic_modelling(
        df,
        data_params.dataset_params.store_path,
        lects_with_topics,
        data_params.dataset_params.topic_modelling)
    vecs = vec.create_vectors_for_lects(
        df, data_params.dataset_params.store_path, data_params.fasttext_params)
    df = sp.split_lects_by_n_grams(df)
    df = freqscore.count_n_grams_frequencies(df)
    df = vec.gather_vector_information(df, vecs)
    return df
