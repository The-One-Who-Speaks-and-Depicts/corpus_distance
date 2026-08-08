import matplotlib
from io import StringIO
import matplotlib.pyplot as plt

import unittest

from Bio import Phylo

from corpus_distance.visualisation.clusterisation import (
    visualise_tree
)

matplotlib.use('Agg')


class TestVisualiseTree(unittest.TestCase):

    def tearDown(self):
        plt.close('all')

    def test_visualise_tree_wrong_cases_data_name(self):

        correct_tree = Phylo.read(
            StringIO(
                "(Novgorod:0.04985,(Polack:0.04279,Smolensk:0.04279)Inner1:0.00706)Inner2:0.00000;"
                ),
                "newick"
                )
        correct_metrics = 'metrics'

        test_cases = [
            {
                "tree": correct_tree,
                "metrics": correct_metrics,
                "data_name": incorrect_arg
            } for incorrect_arg in [
                '',
                [],
                ['string'],
                [['string']],
                {},
                {'string': 1},
                {1: 'string'},
                {'string': 'string'},
                0,
                1,
                0.5,
                False      
            ]]
        for case in test_cases:
            with self.subTest(data=case):
                self.assertRaises(
                    ValueError,
                    visualise_tree,
                    case["tree"],
                    case["metrics"],
                    case["data_name"]
                    )
    
    def test_visualise_tree_wrong_cases_metrics(self):

        correct_tree = Phylo.read(
            StringIO(
                "(Novgorod:0.04985,(Polack:0.04279,Smolensk:0.04279)Inner1:0.00706)Inner2:0.00000;"),
                "newick"
                )
        correct_data_name = 'data'

        test_cases = [
            {
                "tree": correct_tree,
                "metrics": incorrect_arg,
                "data_name": correct_data_name
            } for incorrect_arg in [
                '',
                [],
                ['string'],
                [['string']],
                {},
                {'string': 1},
                {1: 'string'},
                {'string': 'string'},
                0,
                1,
                0.5,
                False
            ]]
        for case in test_cases:
            with self.subTest(data=case):
                self.assertRaises(
                    ValueError,
                    visualise_tree,
                    case["tree"],
                    case["metrics"],
                    case["data_name"]
                    )


    def test_visualise_tree_wrong_cases_tree(self):

        correct_metrics = 'metrics'
        correct_data_name = 'data'

        test_cases = [
            {
                "tree": incorrect_arg,
                "metrics": correct_metrics,
                "data_name": correct_data_name
            } for incorrect_arg in [
                'text',
                '',
                [],
                ['string'],
                [['string']],
                {},
                {'string': 1},
                {1: 'string'},
                {'string': 'string'},
                0,
                1,
                0.5,
                False
            ]]
        for case in test_cases:
            with self.subTest(data=case):
                self.assertRaises(
                    ValueError,
                    visualise_tree,
                    case["tree"], case["metrics"], case["data_name"]
                    )

    
    def test_visualise_tree_correct_case_data_name_in_metrics(self):

        correct_tree = Phylo.read(
            StringIO(
                "(Novgorod:0.04985,(Polack:0.04279,Smolensk:0.04279)Inner1:0.00706)Inner2:0.00000;"
                ),
                "newick"
                )
        correct_metrics = 'metrics for data'
        correct_data_name = 'data'

        fig = visualise_tree(correct_tree, correct_metrics, correct_data_name)

        self.assertEqual(fig.get_suptitle(), correct_metrics)

    def test_visualise_tree_correct_case_data_name_not_in_metrics(self):

        correct_tree = Phylo.read(
            StringIO(
                "(Novgorod:0.04985,(Polack:0.04279,Smolensk:0.04279)Inner1:0.00706)Inner2:0.00000;"
                ),
                "newick"
                )
        correct_metrics = 'metrics'
        correct_data_name = 'Novgorod'
        expected = 'metrics of Novgorod'

        fig = visualise_tree(correct_tree, correct_metrics, correct_data_name)

        self.assertEqual(fig.get_suptitle(), expected)

    def test_visualise_tree_correct_case_check_x(self):

        correct_tree = Phylo.read(
            StringIO(
                "(Novgorod:0.04985,(Polack:0.04279,Smolensk:0.04279)Inner1:0.00706)Inner2:0.00000;"
                ),
                "newick"
                )
        correct_metrics = 'metrics'
        correct_data_name = 'data'

        fig = visualise_tree(correct_tree, correct_metrics, correct_data_name)
        x_label = fig.axes[0].get_xlabel()

        self.assertEqual(x_label, 'branch length')

    def test_visualise_tree_correct_case_check_y(self):

        correct_tree = Phylo.read(
            StringIO(
                "(Novgorod:0.04985,(Polack:0.04279,Smolensk:0.04279)Inner1:0.00706)Inner2:0.00000;"
                ),
                "newick"
                )
        correct_metrics = 'metrics'
        correct_data_name = 'data'

        fig = visualise_tree(correct_tree, correct_metrics, correct_data_name)
        y_label = fig.axes[0].get_ylabel()

        self.assertEqual(y_label, '')

    def test_visualise_tree_correct_case_check_nodes(self):

        correct_tree = Phylo.read(
            StringIO(
                "(Novgorod:0.04985,(Polack:0.04279,Smolensk:0.04279)Inner1:0.00706)Inner2:0.00000;"
                ),
                "newick"
                )
        correct_metrics = 'metrics'
        correct_data_name = 'data'
        terminal_nodes = [node.name for node in correct_tree.get_terminals()]

        fig = visualise_tree(correct_tree, correct_metrics, correct_data_name)
        ax = fig.axes[0]        
        labels = [text.get_text().strip() for text in ax.texts if text.get_text() != ""]

        self.assertTrue(all(label in labels for label in terminal_nodes)) 



if __name__ == '__main__':
    unittest.main()