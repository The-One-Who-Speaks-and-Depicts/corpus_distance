"""
Metrics pipeline module takes assemble data and parameters of the experiment,
and returns the distance measurement values of the lects based on the provided
texts in the dataset
"""

import logging
from os.path import dirname, isdir, realpath
from pandas import DataFrame
from corpus_distance.distance_measurement.analysis import save_data_for_analysis
from corpus_distance.distance_measurement.hybridisation\
    import compare_lects_with_vectors, HybridisationParameters, LectPairInformation
from corpus_distance.cdutils import get_unique_pairs, get_lects_from_dataframe


def score_metrics_for_corpus_dataset(
    df: DataFrame,
    store_path: str = dirname(realpath(__file__)),
    metrics_name: str = "hybrid measurement",
    hybridisation_parameters: HybridisationParameters = HybridisationParameters(),
    ) -> list[tuple[tuple[str,str], int|float]]:
    """
    A function that takes dataset, metrics name and parameters for hybridisation,
    and returns a list of results for each pair of lects in a consecutive order

    Parameters:
        df(DataFrame): dataset with data
        metrics_name(str): name of metrics
        hybridisation_parameters(HybridisationParameters): a set of parameters
        for hybridisation
    Returns:
        overall_results(list[tuple[tuple[str,str], int|float]]): a list of measurements for each
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
        logging.info("Starting scoring %s for %s and %s",
                    metrics_name, i[0], i[1])
        # getting required data for each lect from the dataframe
        lect_1 = list(df[df['lect'] == i[0]]['relative_frequency_n_grams'])[0]
        lect_2 = list(df[df['lect'] == i[1]]['relative_frequency_n_grams'])[0]

        lect_info_1 = list(df[df['lect'] == i[0]]['lect_info'])[0]
        lect_info_2 = list(df[df['lect'] == i[1]]['lect_info'])[0]

        lect_vectors_1 = list(df[df['lect'] == i[0]]['lect_vectors'])[0]
        lect_vectors_2 = list(df[df['lect'] == i[1]]['lect_vectors'])[0]

        lects_for_analysis = LectPairInformation(
            lect_1, lect_2,
            lect_vectors_1, lect_vectors_2,
            lect_info_1, lect_info_2)

        # run metric and save the final results
        analysis_data, result = compare_lects_with_vectors(
            lects_for_analysis,
            hybridisation_parameters
        )
        logging.info("Storing results in %s", store_path)
        save_data_for_analysis(analysis_data, metrics_name, i[0], i[1], store_path)
        logging.info("%s for %s and %s is %s", metrics_name, i[0], i[1], result)
        overall_results.append((i, result))
    logging.info("Resulting distances are %s", overall_results)
    return overall_results
