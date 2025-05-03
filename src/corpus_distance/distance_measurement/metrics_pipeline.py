"""
Metrics pipeline module takes assemble data and parameters of the experiment,
and returns the distance measurement values of the lects based on the provided
texts in the dataset
"""

from os import getcwd
from os.path import isdir
from logging import getLogger, NullHandler
from pandas import DataFrame
from corpus_distance.distance_measurement.analysis\
    import save_data_for_analysis, save_distances_info, MeasurementInfoParams
from corpus_distance.distance_measurement.hybridisation\
    import compare_lects_with_vectors, HybridisationParameters, LectPairInformation
from corpus_distance.cdutils import get_unique_pairs, get_lects_from_dataframe

logger = getLogger(__name__)
logger.addHandler(NullHandler())


def gather_lect_information_from_df(
        df: DataFrame,
        lect_a_name: str, lect_b_name: str) -> LectPairInformation:
    """
    Gets required data for each lect from the provided dataframe,
    using names of the two lects in comparison.

    Arguments:
        df (DataFrame): data frame with all the features, necessary
        (n-shingles; character-based embeddings; alphabet entropy) for the analysis of lects
        lect_a_name (str): name of the first lect in comparison
        lect_b_name (str): name of the second lect in comparison
    
    Results:
        A LectPairInformation class instance that contains tokens,
        alphabet entropy and character-based embeddings for
        each of the two lects under consideration extracted from
        the data frame
    """
    lect_a = list(df[df['lect'] == lect_a_name]['relative_frequency_n_grams'])[0]
    lect_b = list(df[df['lect'] == lect_b_name]['relative_frequency_n_grams'])[0]

    lect_vectors_a = list(df[df['lect'] == lect_a_name]['lect_vectors'])[0]
    lect_vectors_b = list(df[df['lect'] == lect_b_name]['lect_vectors'])[0]

    lect_info_a = list(df[df['lect'] == lect_a_name]['lect_info'])[0]
    lect_info_b = list(df[df['lect'] == lect_b_name]['lect_info'])[0]

    return LectPairInformation(
        lect_a,
        lect_b,
        lect_vectors_a,
        lect_vectors_b,
        lect_info_a,
        lect_info_b
    )


def score_metrics_for_corpus_dataset(
    df: DataFrame,
    dataset_name: str,
    store_path: str = getcwd(),
    metrics_name: str = "hybrid measurement",
    hybridisation_parameters: HybridisationParameters = HybridisationParameters()
    ) -> list[tuple[tuple[str,str], int|float]]:
    """
    A function that takes dataset, metrics name and parameters for hybridisation,
    and returns a list of results for each pair of lects in a consecutive order

    Parameters:
        df (DataFrame): data frame with all the features (n-shingles; character-based embeddings;
        alphabet entropy) for the lects
        metrics_name (str): name of metrics
        hybridisation_parameters(HybridisationParameters): a set of parameters
        for hybridisation
        dataset_name (str): name of dataset
    Returns:
        overall_results (list[tuple[tuple[str,str], int|float]]): a list of measurements for each
        pair of lects in a consecutive order with pair names
    """
    if df is None:
        raise ValueError("No df provided")
    if not isdir(store_path):
        raise ValueError(f'Path {store_path} does not exist')
    # declare arrays
    # calculate distances for each pair of lects
    overall_results = []
    for i in get_unique_pairs(get_lects_from_dataframe(df)):
        logger.info("Starting scoring %s for %s and %s",
            metrics_name, i[0], i[1])

        lects_for_analysis = gather_lect_information_from_df(df, i[0], i[1])

        # run metric and save the final results
        analysis_data, result = compare_lects_with_vectors(
            lects_for_analysis,
            hybridisation_parameters
        )
        logger.info("Storing results in %s", store_path)
        result_info_parameters = MeasurementInfoParams(
            dataset_name, metrics_name, i[0], i[1], store_path
            )
        save_data_for_analysis(analysis_data, result_info_parameters)
        save_distances_info(result, result_info_parameters)
        overall_results.append((i, result))
    logger.info("Resulting distances are %s", overall_results)
    return overall_results
