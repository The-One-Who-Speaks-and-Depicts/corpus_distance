"""
Contains tests for the functions of the corpus_distance.visualistion.shingle_processing module. 
"""
from numpy import array, std, argmin
import matplotlib
import matplotlib.pyplot as plt

import unittest


from corpus_distance.visualisation.shingle_processing import (
    logger, plot_shingle_diversity_barchart, plot_shingle_std
)

from tests.mocks.data_preprocessing import SHINGLE_FREQUENCIES

matplotlib.use('Agg')

class TestPlotShingleDiversityBarchart(unittest.TestCase):
    """
    Contains tests for the plot_shingle_diversity_barchart function.
    """

    def tearDown(self):
        """
        Closes all possible figures.
        """
        plt.close('all')

    def test_plot_shingle_diversity_barchart_bad_cases(self):
        """
        Checks, whether the function correctly processes incorrect arguments.
        """

        for case in [
            {
                "shingle_frequencies": {},
                "bottom": 1,
                "upper": 16,
                "width": 0.5
            },
            {
                "shingle_frequencies": {'lect1': None},
                "bottom": 1,
                "upper": 16,
                "width": 0.5
            },
            {
                "shingle_frequencies": {'lect1': ['1', 1, 0.5]},
                "bottom": 1,
                "upper": 16,
                "width": 0.5
            },
            {
                "shingle_frequencies": {'lect1': [1, 1, 1], 'lect2': [1, 2]},
                "bottom": 1,
                "upper": 16,
                "width": 0.5
            },
            {
                "shingle_frequencies": [],
                "bottom": 1,
                "upper": 16,
                "width": 0.5
            },
            {
                "shingle_frequencies": 0,
                "bottom": 1,
                "upper": 16,
                "width": 0.5
            },
            {
                "shingle_frequencies": 'gf',
                "bottom": 1,
                "upper": 16,
                "width": 0.5
            },
            {
                "shingle_frequencies": SHINGLE_FREQUENCIES,
                "bottom": 0,
                "upper": 16,
                "width": 0.5
            },
            {
                "shingle_frequencies": SHINGLE_FREQUENCIES,
                "bottom": 0.5,
                "upper": 16,
                "width": 0.5
            },
            {
                "shingle_frequencies": SHINGLE_FREQUENCIES,
                "bottom": '1',
                "upper": 16,
                "width": 0.5
            },
            {
                "shingle_frequencies": SHINGLE_FREQUENCIES,
                "bottom": [1],
                "upper": 16,
                "width": 0.5
            },
            {
                "shingle_frequencies": SHINGLE_FREQUENCIES,
                "bottom": {1: 2},
                "upper": 16,
                "width": 0.5
            },
            {
                "shingle_frequencies": SHINGLE_FREQUENCIES,
                "bottom": 1,
                "upper": -1,
                "width": 0.5
            },
            {
                "shingle_frequencies": SHINGLE_FREQUENCIES,
                "bottom": 0,
                "upper": '16',
                "width": 0.5
            },
            {
                "shingle_frequencies": SHINGLE_FREQUENCIES,
                "bottom": 0,
                "upper": False,
                "width": 0.5
            },
            {
                "shingle_frequencies": SHINGLE_FREQUENCIES,
                "bottom": 0,
                "upper": {16: 32},
                "width": 0.5
            },
            {
                "shingle_frequencies": SHINGLE_FREQUENCIES,
                "bottom": 0,
                "upper": [16],
                "width": 0.5
            },
            {
                "shingle_frequencies": SHINGLE_FREQUENCIES,
                "bottom": 3,
                "upper": 2,
                "width": 0.5
            },
            {
                "shingle_frequencies": SHINGLE_FREQUENCIES,
                "bottom": 1,
                "upper": 5,
                "width": '0.5'
            },
            {
                "shingle_frequencies": SHINGLE_FREQUENCIES,
                "bottom": 1,
                "upper": 5,
                "width": [0.5]
            },
            {
                "shingle_frequencies": SHINGLE_FREQUENCIES,
                "bottom": 1,
                "upper": 5,
                "width": {0.5: 0.5}
            },
        ]:
            with self.subTest(data=case):
                self.assertRaises(
                    ValueError,
                    plot_shingle_diversity_barchart,
                    case["shingle_frequencies"],
                    case["bottom"],
                    case["upper"],
                    case["width"]
                )

    def test_plot_shingle_diversity_barchart_bad_cases_non_positive_width(self):
        """
        Checks, whether the function warns in case the width parameter is 0 or less.
        """
        warning = "WARNING:corpus_distance.visualisation.shingle_processing:" \
        "Width (0) is negative or neutral, reconsider setting the parameter"

        with self.assertLogs(logger.name, "WARNING") as captured:
            plot_shingle_diversity_barchart(SHINGLE_FREQUENCIES, 1, 16, 0)

        self.assertTrue(warning in captured.output)


    def test_plot_shingle_diversity_barchart_correct_case(self):
        """
        Checks, whether the function correctly processes the default correct parameters.
        """
        expected = [
            n_shingle_freq for lect in SHINGLE_FREQUENCIES.values() for n_shingle_freq in array(
                lect, dtype=float
                )
            ]

        fig = plot_shingle_diversity_barchart(SHINGLE_FREQUENCIES)
        self.assertEqual(len(fig.axes), 1)
        
        heights = [p.get_height() for p in fig.axes[0].patches]
        self.assertEqual(heights, expected)


class TestPlotShingleSTD(unittest.TestCase):
    """
    Contains tests for the plot_shingle_std function.
    """
    
    def tearDown(self):
        """
        Closes all possible figures.
        """
        plt.close('all')
    
    def test_plot_shingle_std_bad_cases(self):
        """
        Checks, whether the function correctly processes incorrect arguments.
        """
    
        for case in [
            {
                "shingle_frequencies": {},
                "bottom": 1,
                "upper": 16
            },
            {
                "shingle_frequencies": {'lect1': None},
                "bottom": 1,
                "upper": 16
            },
            {
                "shingle_frequencies": {'lect1': ['1', 1, 0.5]},
                "bottom": 1,
                "upper": 16
            },
            {
                "shingle_frequencies": {'lect1': [1, 1, 1], 'lect2': [1, 2]},
                "bottom": 1,
                "upper": 16
            },
            {
                "shingle_frequencies": [],
                "bottom": 1,
                "upper": 16
            },
            {
                "shingle_frequencies": 0,
                "bottom": 1,
                "upper": 16
            },
            {
                "shingle_frequencies": 'gf',
                "bottom": 1,
                "upper": 16
            },
            {
                "shingle_frequencies": SHINGLE_FREQUENCIES,
                "bottom": 0,
                "upper": 16
            },
            {
                "shingle_frequencies": SHINGLE_FREQUENCIES,
                "bottom": 0.5,
                "upper": 16
            },
            {
                "shingle_frequencies": SHINGLE_FREQUENCIES,
                "bottom": '1',
                "upper": 16
            },
            {
                "shingle_frequencies": SHINGLE_FREQUENCIES,
                "bottom": [1],
                "upper": 16
            },
            {
                "shingle_frequencies": SHINGLE_FREQUENCIES,
                "bottom": {1: 2},
                "upper": 16
            },
            {
                "shingle_frequencies": SHINGLE_FREQUENCIES,
                "bottom": 1,
                "upper": -1
            },
            {
                "shingle_frequencies": SHINGLE_FREQUENCIES,
                "bottom": 0,
                "upper": '16'
            },
            {
                "shingle_frequencies": SHINGLE_FREQUENCIES,
                "bottom": 0,
                "upper": False
            },
            {
                "shingle_frequencies": SHINGLE_FREQUENCIES,
                "bottom": 0,
                "upper": {16: 32}
            },
            {
                "shingle_frequencies": SHINGLE_FREQUENCIES,
                "bottom": 0,
                "upper": [16]
            },
            {
                "shingle_frequencies": SHINGLE_FREQUENCIES,
                "bottom": 3,
                "upper": 2
            },
        ]:
            with self.subTest(data=case):
                self.assertRaises(
                    ValueError,
                    plot_shingle_std,
                    case["shingle_frequencies"],
                    case["bottom"],
                    case["upper"]
                )
    
    
    def test_plot_shingle_std_correct_case(self):
        """
        Checks, whether the function correctly processes the default correct parameters.
        """
        expected = []
        bottom_limit = 1
        upper_limit = 16
        for i in range(0, upper_limit - bottom_limit):
            n_quantity = []
            for lect in SHINGLE_FREQUENCIES.keys():
                n_quantity.append(SHINGLE_FREQUENCIES[lect][i])
            expected.append(std(n_quantity))
        expected_min_y = min(expected)
        expected_min_x = range(bottom_limit, upper_limit)[argmin(expected)]
    
        fig = plot_shingle_std(SHINGLE_FREQUENCIES, bottom_limit, upper_limit)
        self.assertEqual(len(fig.axes), 1)

        ax = fig.axes[0]
        lines = ax.get_lines()
        self.assertEqual(len(lines), 18)
        
        line = lines[0]
        _, y_data = line.get_data()
        self.assertEqual(list(y_data), expected)

        marker_line = lines[17]
        x_data, y_data = marker_line.get_data()
        self.assertEqual(x_data[0], expected_min_x)
        self.assertEqual(y_data[0], expected_min_y)
