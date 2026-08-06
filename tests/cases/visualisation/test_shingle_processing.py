"""
Contains tests for the functions of the corpus_distance.visualistion.shingle_processing module. 
"""
from numpy import array
import matplotlib
import matplotlib.pyplot as plt

import unittest


from corpus_distance.visualisation.shingle_processing import (
    logger, plot_shingle_diversity_barchart
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