"""
Topic modelling module helps to clear the text
from the topic words, relevant for particular
documents or document genres.
"""

from copy import deepcopy
from dataclasses import dataclass
from logging import getLogger, NullHandler
from os.path import exists, join

from pandas import DataFrame
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

from corpus_distance.cdutils import clear_stop_words, return_topic_words

logger = getLogger(__name__)
logger.addHandler(NullHandler())

@dataclass
class LDAParams:
    """
    A list of params to provide LDA model with. For
    further information refer to gensim documentation

    Arguments:
    num_topics (int): number of topics to model
    alpha (str): alpha rate
    epochs (int): epochs number
    passes (int): passes for each epoch
    random_state (int): setting random state for reproducibility
    required_topics_num (int): the number of topics to collect
    required_topics_start (int): the first (in a descending order) topic to collect
    """
    num_topics: int = 10
    alpha: str = "auto"
    epochs: int = 300
    passes: int = 500
    random_state: int = 0
    required_topics_num: int | None = None
    required_topics_start: int | None = None


def set_topic_range(params: LDAParams = LDAParams()) -> tuple[int, int]:
    """
    Sets the range of LDA-generated topics in a descending order to collect.

    Arguments:
        params(LDAParams): a dictionary with possible 
        parameters for LdaModel

    Returns:
        A tuple of two integer values,
            first_topic_to_collect: the first (in a descending order) topic to collect
            topic_range_limit: the first topic not to collect
    
    Example:
        If function returns (2, 5), then the selected topics will be 2, 3, 4

    """
    if params.required_topics_start and (
        params.required_topics_start < 0 or \
        params.required_topics_start >= params.num_topics
    ):
        raise ValueError(
            "Incorrect number of the first topic to collect."
            )
    if params.required_topics_num and (
        params.required_topics_num < 0 or \
        params.required_topics_num > params.num_topics or (
            params.required_topics_start and \
            params.required_topics_start + params.required_topics_num > params.num_topics
        )
    ):
        raise ValueError(
            "Incorrect value for range of topics to collect."
        )
    first_topic_to_collect = params.required_topics_start \
            if params.required_topics_start else 0
    topic_range_limit = first_topic_to_collect + params.required_topics_num \
        if params.required_topics_num else params.num_topics

    logger.info(
        "Range of collected topics is %s up to %s",
            str(first_topic_to_collect),
            str(topic_range_limit)
        )

    return (first_topic_to_collect, topic_range_limit)

def build_topic_words_for_lect(
        df: DataFrame, lect: str,
        first_topic: int, topic_range_limit: int,
        params: LDAParams = LDAParams()) -> list[str]:
    """
    Creates a list of topic words, using the texts from a particular lect
    with help of the Latent Dirichlet Association (LDA) technique.

    Arguments:
        df (DataFrame): the data that undergoes topic modelling
        lect (str): a name of lect, for texts of which LDA generates topics
        first_topic (int): the first topic from the ones generated by LDA to collect
        topic_range (int): the number of topics to collect from the ones generated by LDA
        params (LDAParams): a dictionary with possible 
        parameters for LdaModel
    
    Returns:
        lect_topic_words (list[str]): list of strings, each denoting a word from the topics,
        generated for lect by LDA
    """
    logger.info("Building topics for %s lect", lect)

    list_of_texts_split = [
        i.split(' ') for i in list(df[df['lect'] == lect]['text'])
        ]
    common_dictionary = Dictionary(list_of_texts_split)

    common_corpus = [
        common_dictionary.doc2bow(text) for text in list_of_texts_split
        ]

    logger.debug("Modelling %s topics with %s alpha by %s epochs, %s passes; random_state is %s",
                params.num_topics, params.alpha,
                params.epochs, params.passes, params.random_state)

    lda = LdaModel(
        common_corpus,
        num_topics=params.num_topics, alpha=params.alpha,
        iterations=params.epochs, passes=params.passes,
        random_state=params.random_state)

    lect_topic_words = []

    for i in range(
        first_topic, topic_range_limit
        ):
        for j in lda.get_topic_terms(i):
            lect_topic_words.append(common_dictionary[j[0]])

    lect_topic_words = list(sorted(set(lect_topic_words)))

    logger.info(
        "Topics for %s lect are %s", lect, lect_topic_words
        )

    return lect_topic_words



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
    if not isinstance(lects, list) or not all(
        isinstance(i, str) for i in lects
        ):
        raise ValueError("Lects should be a list of strings")
    if not isinstance(params, LDAParams):
        raise ValueError("Params should be of type LDAParams")
    if params.num_topics < 1 or params.epochs < 1 \
        or params.passes < 1:
        raise ValueError(
            "Num_topics, epochs and passes should be positive integers"
            )

    first_topic, last_topic = set_topic_range(params)

    topic_words = {}
    logger.debug("Input checks passed. Topic modelling started.")

    for lect in lects:
        topic_words[lect] = build_topic_words_for_lect(
            df, lect, first_topic, last_topic, params
            )

    return topic_words


def save_topic_modelling_results(
        topic_words: dict, theme_df: DataFrame, output_dir: str
        ) -> None:
    """
    For transparency and reproducibility purposes, the function stores
    the dataframe with both thematic modelling results and original text,
    as well as the dataframe with thematic words by lect,
    in the experiment folder

    Arguments:
        topic_words (dict): a dictionary with lect names as keys, and the lists of 
        topic words as values
        theme_df (DataFrame): a DataFrame with the results of topic modelling
        output_dir (str): the directory where the results are going to be stored
    """
    topic_words_df = DataFrame(
        topic_words.items(), columns = ['lect', 'topic_words']
        )
    topic_words_df.to_csv(
        join(output_dir, 'topic_words_by_lect.csv'),
        index = False
        )
    theme_df.to_csv(
        join(output_dir, 'thematic_modelling_output.csv'),
        index = False
        )

def add_topic_modelling(
    df: DataFrame, output_dir: str,
    topic_words: dict, substitute: str = 'not_substitute') -> DataFrame:
    """
    Enriches the original dataset with texts, 
    stripped off of topic words

    Arguments:
        df (DataFrame): original dataframe with two columns,
        text and lect
        output_dir (str): initial path to directory, where a package will store the results
        topic_words (dict): dictionary with lect names
        (must coincide with lects in df) and
        topic words of their texts,
        assigned respectively
        substitute (str): defines, how model treats topic modelling results. 
        The possible options are:
            * 'substitute' - the model deletes all the topic words from the text
            * 'not_substitute' - the model preserves the text as is
            * 'topic_words_only' - the model preserves only topic words
    Returns:
        theme_df(DataFrame): a deep copy of the original dataframe,
        enriched with text without topic words
    """
    if 'lect' not in df.columns or 'text' not in df.columns:
        raise ValueError("No either \'lect\' or \'text\' columns")
    if not isinstance(topic_words, dict):
        raise ValueError("Topic words should be dictionary")
    if not all(
        isinstance(i, str) for i in topic_words.keys()
        ):
        raise ValueError("The keys of topic words should be strings")
    if not all(
            isinstance(i, list) and all(
                isinstance(j, str) for j in i
                ) for i in topic_words.values()
            ):
        for i in topic_words.values():
            print(i, isinstance(i, list))
            for j in i:
                print(j, isinstance(j, str))
        raise ValueError("The values of topic words should be lists of strings")
    if not exists(output_dir):
        raise ValueError("Output directory does not exist")
    if not isinstance(substitute, str) or substitute not in [
        'not_substitute', 'substitute', 'topic_words_only'
        ]:
        raise ValueError("Substitute should be a string \
        either \'not_substitute\', \'substitute\' or \'topic_words_only\'")
    logger.debug("Input checks passed. Adding thematic modelling.")
    theme_df = deepcopy(df)
    if substitute == 'topic_words_only':
        theme_df['text_topic_normalised'] = theme_df.apply(
        lambda x: return_topic_words(x['text'], topic_words[x['lect']]),
        axis = 1)
    if substitute == 'substitute':
        theme_df['text_topic_normalised'] = theme_df.apply(
            lambda x: clear_stop_words(x['text'], topic_words[x['lect']]),
            axis = 1)
    save_topic_modelling_results(topic_words, theme_df, output_dir)
    if substitute in ['substitute', 'topic_words_only']:
        theme_df['text'] = theme_df['text_topic_normalised']
    logger.debug('Thematic modelling is finished.')
    return theme_df
