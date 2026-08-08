import matplotlib
import matplotlib.pyplot as plt

from numpy import float64
import unittest
from pandas.testing import assert_frame_equal

from pandas import DataFrame

from corpus_distance.visualisation.distance_measurement import plot_clustermap

matplotlib.use('Agg')



class TestPlotClustermap(unittest.TestCase):

    def tearDown(self):
        plt.close('all')

    def test_plot_clustermap_wrong_cases_matrix(self):
        
         
        self.assertRaises(
            ValueError,
            plot_clustermap,
            [[0, 0.3], [0.3, 0, 0.7], [0.6, 0.7, 0]],
            ['A', 'B', 'C']
            )
        
        self.assertRaises(
            ValueError,
            plot_clustermap,
            False,
            ['A', 'B', 'C']
            )
        
        self.assertRaises(
            ValueError,
            plot_clustermap,
            {},
            ['A', 'B', 'C']
            )

        self.assertRaises(
            ValueError,
            plot_clustermap,
            0.5,
            ['A', 'B', 'C']
            )

        self.assertRaises(
            ValueError,
            plot_clustermap,
            0,
            ['A', 'B', 'C']
            )

        self.assertRaises(
            ValueError,
            plot_clustermap,
            '',
            ['A', 'B', 'C']
            )

        self.assertRaises(
            ValueError,
            plot_clustermap,
            [],
            ['A', 'B', 'C']
            )

        self.assertRaises(
            ValueError,
            plot_clustermap,
            [[]],
            ['A', 'B', 'C']
            )

        self.assertRaises(
            ValueError,
            plot_clustermap,
            [[0, 0.3], [0.3, 0, 0.7]],
            ['A', 'B', 'C']
            )

        self.assertRaises(
            ValueError,
            plot_clustermap,
            [[0, 0.3], [0.3, 0, 0.7], [0.6, 0.7, 0]],
            ['A', 'B', 'C']
            )
        
        self.assertRaises(
            ValueError,
            plot_clustermap,
            [[0, 0.3, False], [0.3, 0, 0.7], [False, 0.7, 0]],
            ['A', 'B', 'C']
            )

    def test_plot_clustermap_wrong_case_lects(self):

        self.assertRaises(
            ValueError,
            plot_clustermap,
            [[0, 0.3, 0.6], [0.3, 0, 0.7], [0.6, 0.7, 0]],
            ['A', '', 'C']
            )

        self.assertRaises(
            ValueError,
            plot_clustermap,
            [[0, 0.3, 0.6], [0.3, 0, 0.7], [0.6, 0.7, 0]],
            ['A', 1, 'C']
            )

        self.assertRaises(
            ValueError,
            plot_clustermap,
            [[0, 0.3, 0.6], [0.3, 0, 0.7], [0.6, 0.7, 0]],
            []
            )
        
        self.assertRaises(
            ValueError,
            plot_clustermap,
            [[0, 0.3], [0.3, 0, 0.7], [0.6, 0.7, 0]],
            ''
            )
        
        self.assertRaises(
            ValueError,
            plot_clustermap,
            [[0, 0.3], [0.3, 0, 0.7], [0.6, 0.7, 0]],
            {}
            )
        
        self.assertRaises(
            ValueError,
            plot_clustermap,
            [[0, 0.3], [0.3, 0, 0.7], [0.6, 0.7, 0]],
            0
            )
        
        self.assertRaises(
            ValueError,
            plot_clustermap,
            [[0, 0.3], [0.3, 0, 0.7], [0.6, 0.7, 0]],
            False
            )
        
        self.assertRaises(
            ValueError,
            plot_clustermap,
            [[0, 0.3, 0.6], [0.3, 0, 0.7], [0.6, 0.7, 0]],
            ['A', 'B']
            )

    def test_plot_clustermap_positive_case(self):
        initial_matrix = [[0, float64(0.3), 0.6], [float64(0.3), 0, 0.7], [0.6, 0.7, 0]]
        lects = ['A', 'B', 'C']
        expected = DataFrame(
            initial_matrix, index=lects, columns=lects
            ).sort_index(axis=0).sort_index(axis=1)
        
        fig = plot_clustermap(initial_matrix, lects)
        result = fig.data2d.sort_index(axis=0).sort_index(axis=1)

        assert_frame_equal(expected, result, check_exact=True)

    def test_plot_clustermap_test_cladogram(self):
        """
        Assures the dendrogram existence. As the distances are pre-computed,
        and clustermap itself uses an implementation of a clustering schema
        that itself uses the other one, this test basically checks for the changes
        in the third-party software, in case something goes wrong.
        """
        initial_matrix = [[0, 0.3, 0.6], [0.3, 0, 0.7], [0.6, 0.7, 0]]
        lects = ['A', 'B', 'C']
        expected = {
            'icoord': [
            [15.0, 15.0, 25.0, 25.0],
            [5.0, 5.0, 20.0, 20.0]
            ],
            'dcoord': [
                [0.0, float64(0.3), float64(0.3), 0.0],
                [0.0,
                 float64(0.6499999999999999),
                 float64(0.6499999999999999),
                 float64(0.3)]
                 ],
        }
        
        fig = plot_clustermap(initial_matrix, lects)
        result = {key: fig.dendrogram_col.dendrogram[key] for key in ['icoord', 'dcoord']}

        self.assertEqual(expected, result)



if __name__ == '__main__':
    unittest.main()