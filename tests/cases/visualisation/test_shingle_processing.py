
from numpy import array
import matplotlib
import matplotlib.pyplot as plt

import unittest


from corpus_distance.visualisation.shingle_processing import (
    plot_shingle_diversity_barchart
)

from tests.mocks.data_preprocessing import SHINGLE_FREQUENCIES

matplotlib.use('Agg')

class TestPlotShingleDiversityBarchart(unittest.TestCase):

    def tearDown(self):
        plt.close('all')


    def test_correct_case(self):
        expected = [
            n_shingle_freq for lect in SHINGLE_FREQUENCIES.values() for n_shingle_freq in array(
                lect, dtype=float
                )
            ]

        fig = plot_shingle_diversity_barchart(SHINGLE_FREQUENCIES)
        self.assertEqual(len(fig.axes), 1)
        
        heights = [p.get_height() for p in fig.axes[0].patches]
        self.assertEqual(heights, expected)