"""
Topic modelling module helps to clear the text
from the topic words, relevant for particular
documents or document genres.
"""

from copy import deepcopy
from dataclasses import dataclass
from logging import getLogger, NullHandler, INFO
from os.path import exists, join

from pandas import DataFrame
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

from corpus_distance.cdutils import clear_stop_words, return_topic_words

logger = getLogger(__name__)
logger.setLevel(INFO)
logger.addHandler(NullHandler())

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
    required_topics_num: int | None = None
    required_topics_start: int | None = None



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
    if params.required_topics_num and (
        params.required_topics_num < 0 or \
        params.required_topics_num > params.num_topics \
    ):
        raise ValueError(
            "Incorrect value for range of topics to collect."
        )
    if params.required_topics_start and (
        params.required_topics_start < 0 or \
        params.required_topics_start >= params.num_topics or \
        (
            params.required_topics_num and \
            params.required_topics_start >= params.required_topics_num
        )
    ):
        raise ValueError(
            "Incorrect number of the first topic to collect."
            )
    topic_words = {}
    logger.debug("Input checks passed. Topic modelling started.")
    for lect in lects:
        logger.debug("Building topics for %s lect", lect)
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

        first_topic_to_collect = params.required_topics_start \
            if params.required_topics_start else 0
        range_of_topics_to_collect = params.required_topics_num \
            if params.required_topics_num else params.num_topics

        logger.debug(
            "Range of collected topics for %s lect is %s to %s",
                lect,
                str(first_topic_to_collect),
                str(range_of_topics_to_collect)
            )

        for i in range(
            first_topic_to_collect,
            range_of_topics_to_collect
            ):
            for j in lda.get_topic_terms(i):
                lect_topic_words.append(common_dictionary[j[0]])

        topic_words[lect] = list(set(lect_topic_words))
        logger.debug(
            "Topics for %s lect are %s", lect, str(topic_words[lect])
            )
    return topic_words

def add_thematic_modelling(
    df: DataFrame, output_dir: str,
    topic_words: dict, substitute: str = 'not_substitute') -> DataFrame:
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
    if not isinstance(topic_words, dict) or not all(
        isinstance(i, str) for i in topic_words.keys()
        ) or not all(
            isinstance(i, list) and all(
                isinstance(j, str) for j in i
                ) for i in topic_words.values()
            ):
        raise ValueError("Topic words should be a dictionary with strings \
        as keys and lists of strings as values")
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
    # for transparency and reproducibility purposes, the function stores
    # the dataframe with both thematic modelling results and original text,
    # as well as the dataframe with thematic words by lect,
    # in the experiment folder
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
    if substitute in ['substitute', 'topic_words_only']:
        theme_df['text'] = theme_df['text_topic_normalised']
    logger.debug('Thematic modelling is finished.')
    return theme_df
