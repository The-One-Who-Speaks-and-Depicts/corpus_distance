"""
Topic modelling module helps to clear the text
from the topic words, relevant for particular
documents or document genres.

Functions:

* set_topic_range (params: LDAParams) -> tuple[int, int]: takes 
the set of parameters for topic modelling and returns the range of topics
to undergo selection.
* build_topic_words_for_lect (df: DataFrame, lect: str,
        first_topic: int, topic_range_limit: int,
        params: LDAParams = LDAParams()) -> lect[str]: takes several parameters
to perform the topic modelling of data. The data itself is df, a pandas data frame. 
The lect name, for the texts of which the function performs topic modelling is lect, a string.
first_topic is a positive integer that denotes the sequentially first topic to pick
(topic modelling tools usually sort their topics in terms of crux), topic_range_limit is 
the first topic not to pick. params contains the topic modelling parameters.
Function returns list of topic words, each being a string. 
* get_topic_words_for_lect(df: DataFrame, params: LDAParams()) -> dict:
takes a pandas data frame with columns 'lect' and 'text' in it,
and a set of parameters for topic modelling.
Returns a dictionary that has lects as its keys and the corresponding topic words
as its values
* save_topic_modelling_results(topic_words: dict, theme_df: DataFrame, output_dir: str):
takes a dictionary (with lects as keys and collections of topic words as values),
a data frame (with a column 'text_topics_normalised' in it) and a string (pointing to
the directory where the function should store the results), and saves two .csv files: one
with lects and the corresponding topic words, and the second with the original text for each
lect ('text'), the lect name ('lect') and the results of the application of the topic
modelling results to the original text ('text_topic_normalised').
* add_topic_modelling (df: DataFrame,
topic_words: dict, substitute: str = 'not_substitute') -> DataFrame: takes
a pandas data frame with columns 'lect' and 'text' in it,
a dictionary (with lects as keys and collections of topic words as values), and
a string, pointing to the type of adding topic modelling to the dataset: only storing
its results ('not_substitute'), replacing the original text with its results ('topic_words_only'),
or removing its results from the original text ('substitute'). It passes further
the data frame with the results (or lack thereof) in a column 'text_topic_normalised'. 
"""

from copy import deepcopy
from dataclasses import dataclass
from logging import getLogger, NullHandler
from os.path import exists, join

from pandas import DataFrame
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

from corpus_distance.cdutils import clear_stop_words, return_topic_words, get_lects_from_dataframe

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
    logger.debug("Input params: %s", locals())
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
    logger.debug("Input params: %s", locals())

    if not isinstance(df, DataFrame) or 'lect' not in df.columns or 'text' not in df.columns:
        raise ValueError(f"df should be a data frame that has column lect in it, received {df}")

    if not isinstance(params, LDAParams):
        raise ValueError(f"The params for LDA should be an LDAParams object, received {params}")

    logger.info("Building topics for %s lect", lect)

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
    df: DataFrame, params: LDAParams = LDAParams()) -> dict:
    """
    Takes text in each lect within the given datasets
    to return topic words for each given lect
    with LDA model. 

    Arguments:
        df(DataFrame): a dataframe with texts and lects
        params(LDAParams): a dictionary with possible 
        parameters for LdaModel
        
    Returns:
        topic_words(dict): dictionary with lects as keys,
        and topic words for the texts as values
    """
    logger.debug("Input params: %s", locals())
    if not isinstance(df, DataFrame) or 'lect' not in df.columns or 'text' not in df.columns:
        raise ValueError("No either \'lect\' or \'text\' columns")
    if not isinstance(params, LDAParams):
        raise ValueError("Params should be of type LDAParams")
    if params.num_topics < 1 or params.epochs < 1 \
        or params.passes < 1:
        raise ValueError(
            "Num_topics, epochs and passes should be positive integers"
            )

    first_topic, last_topic = set_topic_range(params)

    topic_words = {}

    lects = get_lects_from_dataframe(df)

    for lect in lects:
        topic_words[lect] = build_topic_words_for_lect(
            df, lect, first_topic, last_topic, params
            )
    logger.debug("Result: %s", topic_words)
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
    logger.debug("Input params: %s", locals())
    if not isinstance(theme_df, DataFrame) or list(
        set(theme_df.columns)
        ) != list(set(['lect', 'text', 'text_topic_normalised'])):
        raise ValueError("theme_df should be a pandas DataFrame with" \
                          "columns \'lect\', \'text\' and \'text_topic_normalised\', " \
                          f"received {theme_df}")
    if not isinstance(topic_words, dict) or list(set(
        topic_words.keys()
    )) != list(
        set(theme_df['lect'].unique())
        ):
        raise ValueError("topic_words should be a dictionary with the same lects as keys" \
        f" as values in the theme_df \'lect\' column, received {topic_words}")
    if not isinstance(output_dir, str) or not exists(output_dir):
        raise ValueError("output_dir should be a path to an existing directory, " \
                         f"received {output_dir}")
    topic_words_df = DataFrame(
        topic_words.items(), columns = ['lect', 'topic_words']
        ).set_index('lect')
    topic_words_df.to_csv(
        join(output_dir, 'topic_words_by_lect.csv'),
        index_label='lect'
        )
    theme_df.to_csv(
        join(output_dir, 'thematic_modelling_output.csv'),
        index = False
        )

def add_topic_modelling(
    df: DataFrame,
    topic_words: dict, substitute: str = 'not_substitute') -> DataFrame:
    """
    Enriches the original dataset with texts, 
    stripped off of topic words.

    Arguments:
        df (DataFrame): original dataframe with two columns,
        text and lect
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
    logger.debug("Input params: %s", locals())
    if not isinstance(df, DataFrame) or 'lect' not in df.columns or 'text' not in df.columns:
        raise ValueError("No either \'lect\' or \'text\' columns")
    if not isinstance(topic_words, dict) or not all(
        isinstance(i, str) for i in topic_words.keys()
        ):
        raise ValueError("The keys of topic words should be strings")
    if not all(
            isinstance(i, list) and all(
                isinstance(j, str) for j in i
                ) for i in topic_words.values()
            ):
        raise ValueError("The values of topic words should be lists of strings")
    if list(set(
        topic_words.keys()
    )) != list(
        set(df['lect'].unique())
        ):
        raise ValueError("topic_words should be a dictionary with the same lects as keys" \
        f" as values in the theme_df \'lect\' column, received {topic_words}")
    if not isinstance(substitute, str) or substitute not in [
        'not_substitute', 'substitute', 'topic_words_only'
        ]:
        raise ValueError("Substitute should be a string \
        either \'not_substitute\', \'substitute\' or \'topic_words_only\'")
    theme_df = deepcopy(df)
    match substitute:
        case 'topic_words_only':
            theme_df['text_topic_normalised'] = theme_df.apply(
                lambda x: return_topic_words(x['text'], topic_words[x['lect']]),
                axis = 1)
        case 'substitute':
            theme_df['text_topic_normalised'] = theme_df.apply(
                lambda x: clear_stop_words(x['text'], topic_words[x['lect']]),
                axis = 1)
        case 'not_substitute':
            theme_df['text_topic_normalised'] = theme_df['text']
    if substitute in ['substitute', 'topic_words_only']:
        theme_df['text'] = theme_df['text_topic_normalised']
    logger.debug('Result: %s', theme_df)
    return theme_df
