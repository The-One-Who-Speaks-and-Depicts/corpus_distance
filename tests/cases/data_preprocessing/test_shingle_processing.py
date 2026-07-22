"""
Tests for shingle processing module.
"""

import unittest

import pandas as pd

from corpus_distance.data_preprocessing import shingle_processing as shm

class TestSplitIntoNShingles(unittest.TestCase):
    """
    Tests function n_gram_split.
    """

    def test_error_cases(self):
        """
        Checks data of wrong type and incorrect values.
        """
        self.assertRaises(AttributeError, shm.n_gram_split, 123)
        self.assertRaises(AttributeError, shm.n_gram_split, [])
        self.assertRaises(AttributeError, shm.n_gram_split, False)
        self.assertRaises(AttributeError, shm.n_gram_split, {})
        self.assertRaises(AttributeError, shm.n_gram_split, {'123': '456'})

        self.assertRaises(ValueError, shm.n_gram_split, 'fdgdfg', -1)
        self.assertRaises(ValueError, shm.n_gram_split, 'fdgdfg', 0)
        self.assertRaises(ValueError, shm.n_gram_split, 'fdgdfg', 2.5)
        self.assertRaises(ValueError, shm.n_gram_split, 'fdgdfg', '123')
        self.assertRaises(ValueError, shm.n_gram_split, 'fdgdfg', False)

    def test_positive_cases_single_token(self):
        """
        Checks the correct data, if there is only one token.
        """

        self.assertEqual(shm.n_gram_split('fdgfdg', 1), ['^', 'f', 'd', 'g', 'f', 'd', 'g', '$'])
        self.assertEqual(shm.n_gram_split('fdgfdg', 2), ['^f', 'fd', 'dg', 'gf', 'fd', 'dg', 'g$'])
        self.assertEqual(shm.n_gram_split('fdgfdg', 3), ['^fd', 'fdg', 'dgf', 'gfd', 'fdg', 'dg$'])

    def test_positive_case_with_weird_symbol(self):
        """
        Checks the deletion of a void symbol that often appears in decoded text data.
        """

        test_string = f'{chr(65279)}f'

        self.assertEqual(shm.n_gram_split(test_string, 1), ['^', 'f', '$'])



    def test_positive_cases_multiple_tokens(self):
        """
        Checks the correct data, if there is more than one token.
        """

        self.assertEqual(
            shm.n_gram_split('f dgfdg', 1),
            ['^', 'f', '$', '^', 'd', 'g', 'f', 'd', 'g', '$']
            )
        self.assertEqual(
            shm.n_gram_split('f dgfdg', 2),
            ['^f', 'f$', '^d', 'dg', 'gf', 'fd', 'dg', 'g$']
            )
        self.assertEqual(
            shm.n_gram_split('f dgfdg', 3),
            ['^f$', '^dg', 'dgf', 'gfd', 'fdg', 'dg$']
            )
        self.assertEqual(shm.n_gram_split('f dgfdg', 4), ['^f$', '^dgf', 'dgfd', 'gfdg', 'fdg$'])
        self.assertEqual(shm.n_gram_split('fd gfdg', 4), ['^fd$', '^gfd', 'gfdg', 'fdg$'])


class TestAssigningNGramsToLects(unittest.TestCase):
    """
    Tests function assign_n_grams_to_lects.
    """

    def test_no_suitable_columns(self):
        """
            Checks the dataframe with not a single suitable column.
        """
        data = {"1": ["fdg", "cdg"], "2": ["abc", "cdg"]}

        data_frame = pd.DataFrame.from_dict(data, orient="index", columns=["test", "lest"])

        self.assertRaises(ValueError, shm.assign_n_grams_to_lects, data_frame)

    def test_one_correct_column(self):
        """
            Checks the dataframe with a single correct column.
        """
        data = {"1": ["fdg", "cdg"], "2": ["abc", "cdg"]}

        data_frame = pd.DataFrame.from_dict(data, orient="index", columns=["text", "lect"])

        self.assertRaises(ValueError, shm.assign_n_grams_to_lects, data_frame)

    def test_correct_params(self):
        """
            Checks whether the correct params yield correct results.
        """
        data =  {"0": ["Smol", ["^f$","^dgf","dgfd","gfdg","fdg$"]],
                      "1": ["Pol", ["^fd$","^gfd","gfdg","fdg$"]],
                      "2": ["Pol", ["^f$"]]}
        data_frame = pd.DataFrame.from_dict(data, orient="index", columns=["lect", "n_grams"])
        result_dict = {
            "Smol": ["^f$","^dgf","dgfd","gfdg","fdg$"],
            "Pol": ["^fd$","^gfd","gfdg","fdg$", "^f$"]
        }

        n_grams_by_lects = shm.assign_n_grams_to_lects(data_frame)

        self.assertEqual(n_grams_by_lects, result_dict)


class TestSplitDfIntoNShigles(unittest.TestCase):
    """
        Tests function split_lects_by_n_grams.
    """

    def test_no_suitable_columns(self):
        """
            Checks the dataframe with not a single suitable column.
        """
        data = {"1": ["fdg", "cdg"], "2": ["abc", "cdg"]}

        data_frame = pd.DataFrame.from_dict(data, orient="index", columns=["test", "lest"])

        self.assertRaises(ValueError, shm.split_lects_by_n_grams, data_frame)

    def test_one_suitable_column(self):
        """
            Checks the dataframe with a single suitable column.
        """
        data = {"1": ["fdg", "cdg"], "2": ["abc", "cdg"]}

        data_frame = pd.DataFrame.from_dict(data, orient="index", columns=["text", "lest"])

        self.assertRaises(ValueError, shm.split_lects_by_n_grams, data_frame)

    def test_correct_params(self):
        """
            Checks whether the correct params yield correct values.
        """
        data = {"0": ["f dgfdg", "Smol"], "1": ["fd gfdg", "Pol"]}
        eval_data =  {"0": ["Smol", ["^f$","^dgf","dgfd","gfdg","fdg$"]],
                      "1": ["Pol", ["^fd$","^gfd","gfdg","fdg$"]]}
        data_frame = pd.DataFrame.from_dict(data, orient="index", columns=["text", "lect"])
        eval_df = pd.DataFrame.from_dict(eval_data, orient="index", columns=["lect", "n_grams"])

        result_df = shm.split_lects_by_n_grams(data_frame, 4)

        # transforming to dict to delete hidden pandas metadata
        # that do not influence final results
        self.assertEqual(result_df.to_dict(orient="records"), eval_df.to_dict(orient="records"))
