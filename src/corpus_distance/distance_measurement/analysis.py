"""
Analysis module provides data for further qualitative investigation
into the results of the metrics inner workings.
"""
from dataclasses import dataclass

from os.path import isdir, join

from logging import getLogger, NullHandler

from pandas import DataFrame

logger = getLogger(__name__)
logger.addHandler(NullHandler)

@dataclass
class MeasurementInfoParams:
    """
    Contains data required to provide the information
    about the distance measurement between two lects.

    Arguments:
        dataset_name (str): the name of the dataset
        metrics_name (str): the name of the metrics
        lect_a_name (str): the name of the first lect in comparison
        lect_b_name (str): the name of the second lect in comparison
        store_path (str): the folder where the function should save the results of comparison
    """
    dataset_name: str
    metrics_name: str
    lect_a_name: str
    lect_b_name: str
    store_path: str


def save_distances_info(
        result: int | float,
        params: MeasurementInfoParams) -> None:
    """
    Stores the distance value between the two lects in comparison.

    Arguments:
        result (int | float): the distance value
        params (MeasurementInfoParams): the information about the measurement,
        including names of the compared lects, the dataset name, the metric name,
        and the folder where the function stores comparison results.
    """
    if not isdir(params.store_path):
        raise ValueError(f'Path {params.store_path} does not exist')
    with open(
        join(params.store_path, params.metrics_name + ".info"), "a", encoding="utf-8"
        ) as f:
        distance_info = [
            params.dataset_name, params.metrics_name,
            params.lect_a_name, params.lect_b_name,
            str(result)
            ]
        f.write('\t'.join(distance_info) + "\n")
    logger.debug("Distances for %s and %s stored", params.lect_a_name, params.lect_b_name)

def save_data_for_analysis(
        data_for_analysis: tuple[dict, dict],
        params: MeasurementInfoParams) -> None:
    """
    Takes results of measurements and saves them to .csv file.

    Parameters:
        data_for_analysis(tuple[dict, dict]): dicts with n-grams as keys, 
        and tuples of other n-grams and metric as values for both DistRank
        and hybrid
        params (MeasurementInfoParams): the information about the measurement,
        including names of the compared lects, the dataset name, the metric name,
        and the folder where the function stores comparison results.
    """
    if not isdir(params.store_path):
        raise ValueError(f'Path {params.store_path} does not exist')
    errors_list = []
    if data_for_analysis[0]:
        for i in data_for_analysis[0].keys():
            errors_list.append(
                [i, "id.", params.metrics_name + " - DistRank", data_for_analysis[0][i]]
                )
    if data_for_analysis[1]:
    # as two first columns are lect, for their respective arrays first two
    # columns should be filled correspondingly
        if data_for_analysis[1][0]:
            # recording information for each n-gram
            for i in data_for_analysis[1][0].keys():
                # recording information for each n-gram of other lect that has
                # minimal distance with this n-gram
                for j in data_for_analysis[1][0][i]:
                    other_lect_n_gram = j[0]
                    distance = j[1]
                    errors_list.append([i, other_lect_n_gram, params.metrics_name + " - hybrid",
                                        distance])
        if data_for_analysis[1][1]:
            for i in data_for_analysis[1][1].keys():
                for j in data_for_analysis[1][1][i]:
                    other_lect_n_gram = j[0]
                    distance = j[1]
                    errors_list.append([other_lect_n_gram, i, params.metrics_name + " - hybrid",
                                        distance])
    errors_data = DataFrame(
        errors_list, columns=[
            params.lect_a_name, params.lect_b_name, params.metrics_name, "Distance"
            ]
        )
    errors_data.to_csv(
        join(
            params.store_path,
            params.metrics_name + "_" + params.lect_a_name + "_" + params.lect_b_name + ".csv"
            ), index=False
        )
    logger.debug("Data for analysis of %s and %s stored", params.lect_a_name, params.lect_b_name)
