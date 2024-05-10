"""
Analysis module provides data for further qualitative investigation
into the results of the metrics inner workings.
"""

from os.path import dirname, isdir, join, realpath
from pandas import DataFrame

def save_data_for_analysis(
        data_for_analysis: tuple[dict, dict],
        metrics_name: str,
        lect_a_name: str,
        lect_b_name: str,
        store_path: str = dirname(realpath(__file__))) -> None:
    """
    Takes results of measurements and saves them to .csv file.

    Parameters:
        data_for_analysis(tuple[dict, dict]): dicts with n-grams as keys, 
        and tuples of other n-grams and metric as values for both DistRank
        and hybrid
        metrics_name(str): name of metrics
        lect_a_name(str): name of the first lect
        lect_b_name(str): name of the second lect
        store_path(str): a path to store data
    """
    if not isdir(store_path):
        raise ValueError(f'Path {store_path} does not exist')
    errors_list = []
    if data_for_analysis[0]:
        for i in data_for_analysis[0].keys():
            errors_list.append([i, "id.", metrics_name + " - DistRank", data_for_analysis[0][i]])
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
                    errors_list.append([i, other_lect_n_gram, metrics_name + " - hybrid",
                                        distance])
        if data_for_analysis[1][1]:
            for i in data_for_analysis[1][1].keys():
                for j in data_for_analysis[1][1][i]:
                    other_lect_n_gram = j[0]
                    distance = j[1]
                    errors_list.append([other_lect_n_gram, i, metrics_name + " - hybrid",
                                        distance])
    errors_data = DataFrame(
        errors_list, columns=[lect_a_name, lect_b_name, metrics_name, "Distance"]
        )
    errors_data.to_csv(
        join(
            store_path, metrics_name + "_" + lect_a_name + "_" + lect_b_name + ".csv"
            ), index=False
        )
