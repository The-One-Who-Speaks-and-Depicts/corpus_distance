"""
Topic modelling module helps to clear the text
from the topic words, relevant for particular
documents or document genres.
"""

from copy import deepcopy
from dataclasses import dataclass

from pandas import DataFrame
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

from corpus_distance.cdutils import clear_stop_words

@dataclass
class LDAParams:
    """
    A list of params to provide LDA model with. For
    further information refer to gensim documentation

    Arguments:
    num_topics(int): number of topics to model
    alpha(str): alpha rate
    epochs(int): epochs number
    passes(int): passes for each epoch
    """
    num_topics: int = 10
    alpha: str = "auto"
    epochs: int = 300
    passes: int = 500

def get_topic_words_for_lects(
    df: DataFrame, lects: list[str],
    params: LDAParams = LDAParams()) -> dict:
    """
    Takes text in each lect within the given datasets
    to return topic words for each given lect
    with LDA model. 

    Arguments:
        df(DataFrame): a dataframe with texts and lects
        lects(list[str]): a set of lects
        params(LDAParams): a dictionary with possible 
        parameters for LdaModel
        
    Returns:
        topic_words(dict): dictionary with lects as keys,
        and topic words for the texts as values
    """
    if 'lect' not in df.columns or 'text' not in df.columns:
        raise ValueError("No either \'lect\' or \'text\' columns")
    topic_words = {}
    for lect in lects:
        list_of_texts_split = [
            i.split(' ') for i in list(df[df['lect'] == lect]['text'])
            ]
        common_dictionary = Dictionary(list_of_texts_split)

        common_corpus = [
            common_dictionary.doc2bow(text) for text in list_of_texts_split
            ]

        lda = LdaModel(
            common_corpus,
            num_topics=params.num_topics, alpha=params.alpha,
            iterations=params.epochs, passes=params.passes)

        lect_topic_words = []

        for i in range(params.num_topics):
            for j in lda.get_topic_terms(i):
                lect_topic_words.append(common_dictionary[j[0]])

        topic_words[lect] = list(set(lect_topic_words))
    return topic_words

def add_thematic_modelling(
    df: DataFrame,
    topic_words: dict, substitute: bool = False) -> DataFrame:
    """
    Enriches the original dataset with texts, 
    stripped off of topic words

    Arguments:
        df(DataFrame): original dataframe with two columns,
        text and lect
        topic_words(dict): dictionary with lect names
        (must coincide with lects in df) and
        topic words of their texts,
        assigned respectively
        substitute(bool): whether a text without stop words
        substitutes the original, or not
    Returns:
        theme_df(DataFrame): a deep copy of the original dataframe,
        enriched with text without topic words
    """
    if 'lect' not in df.columns or 'text' not in df.columns:
        raise ValueError("No either \'lect\' or \'text\' columns")
    theme_df = deepcopy(df)
    theme_df['text_topic_normalised'] = theme_df.apply(
        lambda x: clear_stop_words(x['text'], topic_words[x['lect']]),
        axis = 1)
    if substitute:
        theme_df['text'] = theme_df['text_topic_normalised']
    return theme_df
