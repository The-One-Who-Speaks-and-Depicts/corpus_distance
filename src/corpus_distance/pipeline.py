"""
Pipeline module streamlines data import and preprocessing, as well as
distance measurement and clusterisation of lects
"""
from dataclasses import dataclass
import importlib
import json
from os import mkdir
from os.path import isdir, dirname, realpath
import logging
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from corpus_distance.cdutils import get_lects_from_dataframe
from corpus_distance.data_preprocessing.data_pipeline import assemble_dataset
from corpus_distance.data_preprocessing.topic_modelling import LDAParams
from corpus_distance.data_preprocessing.vectorisation import FastTextParams
from corpus_distance.distance_measurement.hybridisation import HybridisationParameters
from corpus_distance.clusterisation.clusterisation import ClusterisationParameters
from corpus_distance.distance_measurement.metrics_pipeline import score_metrics_for_corpus_dataset
from corpus_distance.clusterisation.clusterisation import clusterise_lects_from_distance_matrix
from corpus_distance.data.data_resources import config


@dataclass
class DataParameters:
    """
    Class with parameters that define data loading and preprocessing
    part of pipeline

    Parameters:
        content_path(str): path to directory with text files; text files should be
        named as "TEXT.LECT.txt", and consist of tokenised texts, transformed into
        a single string
        split(int|float): a number from 0 to 1, which signals, which percentage of
        source data should form the basis for clusterisation
        lda_params(LDAParams): a set of parameters for latent dirichlet association
        model of gensim package, for details see LDAParams documentation
        topic_modelling(bool): flag that describes the choice of user to change original text
        to text without topic words
        fasttext_params(FastTextParams): a set of parameters for FastText model that
        builds symbol vectors, for details see FastText documentation
    """
    content_path: str = "default"
    split: int | float = 1
    lda_params: LDAParams = LDAParams()
    topic_modelling: bool = False
    fasttext_params: FastTextParams = FastTextParams()



@dataclass
class ConfigurationParameters:
    """
    Class with all the required parameters for pipeline running.

    Parameters:        
        store_path(str): path to directory, where program will put the results
        data_params(DataParameters): settings for data loading and preprocessing
        hybridisation_parameters(HybridisationParameters): settings for
        hybridisation metrics, for details see HybridisationParams documentation
        metrics_name(str): name of hybridisation parameters and data_name combination,
        either collected automatically, or input by user
        clusterisation_parameters(ClusterisationParameters): settings for 
        clusterisation, for details see ClusterisationParameters documentation
    """

    store_path: str = "default"
    data_params: DataParameters = DataParameters()
    hybridisation_parameters: HybridisationParameters = HybridisationParameters()
    metrics_name: str = "default_metrics_name"
    clusterisation_parameters: ClusterisationParameters = ClusterisationParameters()



def set_storage_directory(store_path: str) -> str:
    """
    Sets directory for experiment results, in case of its absence,
    creates it

    Parameters:
        store_path(str): initial path to directory, where a program will put the results
    Returns:
        store_path(str): final path to directory, where a program will put the results
    """
    if (store_path and store_path != "default"):
        if not isdir(store_path):
            logging.info("Directory %s not exists, creating directory", store_path)
            mkdir(store_path)
        logging.info("Directory set to %s", store_path)
        return store_path
    return dirname(realpath(__file__))



def set_lda_params(lda_cfg: dict) -> LDAParams:
    """
    Creates a default LDAParams object and 
    alters it, if user provides any kind of
    specific information

    Parameters:
        lda_cfg(dict): user-provided parameters for 
        LDA
    Returns:
        lda_params(LDAParams): full set of LDAParams for the model to train on 
    """
    lda_params = LDAParams()
    if lda_cfg["num_topics"]:
        lda_params.num_topics = lda_cfg["num_topics"]
    if lda_cfg["alpha"]:
        lda_params.alpha = lda_cfg["alpha"]
    if lda_cfg["epochs"]:
        lda_params.epochs = lda_cfg["epochs"]
    if lda_cfg["passes"]:
        lda_params.passes = lda_cfg["passes"]
    return lda_params



def set_fast_text_params(fasttext_cfg: dict) -> FastTextParams:
    """
    Creates a default FastTextParams object and
    alters it, if user provides any kind of
    specific information

    Parameters:
        fasttext_cfg(dict): user-provided parameters for 
        FastText embeddings training
    Returns:
        fasttext_params(FastTextParams): full set of FastTextParams for the model to train on
    """
    fasttext_params = FastTextParams()
    if fasttext_cfg["vector_size"]:
        fasttext_params.vector_size = fasttext_cfg["vector_size"]
    if fasttext_cfg["window"]:
        fasttext_params.window = fasttext_cfg["window"]
    if fasttext_cfg["min_count"]:
        fasttext_params.min_count = fasttext_cfg["min_count"]
    if fasttext_cfg["workers"]:
        fasttext_params.workers = fasttext_cfg["workers"]
    if fasttext_cfg["epochs"]:
        fasttext_params.epochs = fasttext_cfg["epochs"]
    if fasttext_cfg["seed"]:
        fasttext_params.seed = fasttext_cfg["seed"]
    if fasttext_cfg["sg"]:
        fasttext_params.sg = fasttext_cfg["sg"]
    return fasttext_params



def set_data_configuration(data_cfg: dict) -> DataParameters:
    """
    Creates a default DataParameters object and
    alters it, if user provides any kind of
    specific information

    Parameters:
        data_cfg(dict): user-provided parameters for 
        data loading and preprocessing
    Returns:
        data_params(DataParameters): full set of DataParameters for the model to train on
    """
    data_params = DataParameters()
    if (data_cfg["content_path"] and data_cfg["content_path"] != "default"):
        data_params.content_path = data_cfg["content_path"]
    if data_cfg["split"]:
        data_params.split = data_cfg["split"]
    if data_cfg["lda_params"]:
        data_params.lda_params = set_lda_params(data_cfg["lda_params"])
    if "topic_modelling" in data_cfg.keys():
        data_params.topic_modelling = data_cfg["topic_modelling"]
    if data_cfg["fasttext_params"]:
        data_params.fasttext_params = set_fast_text_params(data_cfg["fasttext_params"])
    return data_params



def set_hybrid_configuration(hybrid_cfg: dict) -> HybridisationParameters:
    """
    Creates a default HybridisationParameters object and
    alters it, if user provides any kind of
    specific information

    Parameters:
        hybrid_cfg(dict): user-provided parameters for 
        hybridisation between frequency and string similarity-based metrics
    Returns:
        hybrid_params(HybridisationParams): full set of HybridisationParameters
        for the methods to score distance
    """
    hybrid_params = HybridisationParameters()
    if "soerensen" in hybrid_cfg.keys():
        hybrid_params.soerensen = hybrid_cfg["soerensen"]
    if "hybridisation" in hybrid_cfg.keys():
        hybrid_params.hybridisation = hybrid_cfg["hybridisation"]
    if "hybridisation_as_array" in hybrid_cfg.keys():
        hybrid_params.hybridisation_as_array = hybrid_cfg["hybridisation_as_array"]
    if hybrid_cfg["metrics"]:
        function_string = hybrid_cfg["metrics"]
        mod_name, func_name = function_string.rsplit('.',1)
        mod = importlib.import_module(mod_name)
        hybrid_params.metrics = getattr(mod, func_name)
    if "alphabet_normalisation" in hybrid_cfg.keys():
        hybrid_params.alphabet_normalisation = hybrid_cfg["alphabet_normalisation"]
    return hybrid_params



def set_metrics_name(
        cfg: dict,
        hybridisation_parameters: HybridisationParameters):
    """
    Combines user-defined hybridisation parameters into a metrics name,
    or uses a user-provided one

    Parameters:
        cfg(dict): a full set of user-defined configuration parameters
        hybridisation_parameters(HybridisationParameters): a combined 
        set of default and user-defined hybridisation parameters that
        reflect the hybridisation flow
    Returns:
        metrics_name(str): a final metrics name
    """
    if (cfg["metrics_name"] and cfg["metrics_name"] != "default_metrics_name"):
        return cfg["metrics_name"]
    metrics_name = ""
    if cfg["clusterisation_parameters"]["data_name"]:
        metrics_name += f'{cfg["clusterisation_parameters"]["data_name"]}-'
    if cfg["data"]["split"]:
        metrics_name += f'{str(cfg["data"]["split"])}-'
    metrics_name += f'{str(cfg["data"]["topic_modelling"])}-'
    metrics_name += "DistRank-"
    metrics_name += str(hybridisation_parameters)
    return metrics_name



def set_clusterisation_parameters(clust_cfg: dict,
                                  metrics_name: str,
                                  store_path: str) -> ClusterisationParameters:
    """
    Creates a default ClusterisationParameters object and
    alters it, if user provides any kind of
    specific information

    Parameters:
        clust_cfg(dict): user-provided parameters for 
        clusterisation of lects
        metrics_name(str): a name of metrics that scores distances for further classification
        store_path(str): a folder for results storage
    Returns:
        fasttext_params(ClusterisationParameters): full set of ClusterisationParameters
        for the package clusterisation
    """
    clust_params = ClusterisationParameters()
    if clust_cfg["data_name"]:
        clust_params.data_name = clust_cfg["data_name"]
    if clust_cfg["outgroup"]:
        clust_params.outgroup = clust_cfg["outgroup"]
    clust_params.metrics = metrics_name
    clust_params.store_path = store_path
    if clust_cfg["classification_method"]:
        if clust_cfg["classification_method"] not in ["upgma", "nj"]:
            raise ValueError("Only UPGMA and NJ classifiers are available")
        if clust_cfg["classification_method"] == "upgma":
            clust_params.classification_method = DistanceTreeConstructor().upgma
        if clust_cfg["classification_method"] == "nj":
            clust_params.classification_method = DistanceTreeConstructor().nj
    return clust_params



def set_configuration(cfg: dict) -> ConfigurationParameters:
    """
    Takes dict with configuration values and returns object with the user-defined or 
    default parameters, which is then passed to the pipeline.

    Parameters:
        cfg(dict): dictionary with the user-defined parameters
    Returns:
        cfg_params(ConfigurationParameters): object containig the required parameters,
        joining user input with default parameters, if necessary
    """
    cfg_params = ConfigurationParameters()
    cfg_params.store_path = set_storage_directory(cfg["store_path"])
    if cfg["data"]:
        cfg_params.data_params = set_data_configuration(cfg["data"])
    if cfg["hybridisation_parameters"]:
        cfg_params.hybridisation_parameters =\
            set_hybrid_configuration(cfg["hybridisation_parameters"])
    cfg_params.metrics_name = set_metrics_name(cfg, cfg_params.hybridisation_parameters)
    if cfg["clusterisation_parameters"]:
        cfg_params.clusterisation_parameters =\
            set_clusterisation_parameters(
                cfg["clusterisation_parameters"],
                cfg_params.metrics_name,
                cfg_params.store_path
                )
    return cfg_params



def perform_clusterisation(config_path: str = 'default') -> None:
    """
    Takes path to configuration and performs clusterisation

    Parameters:
        config_path(str): path to json file with the required data
    """
    if config_path == 'default':
        cfg = set_configuration(config)
    else:
        with open(config_path, 'r', encoding='utf-8') as inp:
            cfg = set_configuration(json.load(inp))
    logging.info('Configuration set')
    logging.info("Reading data")
    data = assemble_dataset() if cfg.data_params.content_path == 'default' else assemble_dataset(
        cfg.data_params.content_path,
        cfg.data_params.split,
        cfg.data_params.lda_params,
        cfg.data_params.topic_modelling,
        cfg.data_params.fasttext_params
        )
    logging.info("Scoring distances")
    distances = score_metrics_for_corpus_dataset(
        data,
        cfg.store_path,
        cfg.metrics_name,
        cfg.hybridisation_parameters
        )
    cfg.clusterisation_parameters.lects = get_lects_from_dataframe(data)
    logging.info("Initialising clusterisation")
    clusterise_lects_from_distance_matrix(
        distances,
        cfg.clusterisation_parameters
        )
    logging.info("Clusterisation performed")
