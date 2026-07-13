"""
Tests for the data loading module.
"""

import unittest
from os import mkdir
from os.path import join

from shutil import rmtree
import pandas as pd

from corpus_distance.data_preprocessing import data_loading as dl

class TestLoadDefaultData(unittest.TestCase):
    """
    Tests the correct call of load_default_data.
    """
    def setUp(self):
        """
        Loads the default data.
        """
        self.data = dl.load_default_data()

    def tearDown(self):
        """
        Deletes the default data to save memory.
        """
        del self.data

    def test_load_default_data_check_type(self):
        """
        Checks, whether the loaded data is a data frame.
        """
        self.assertIsInstance(self.data, pd.DataFrame)

    def test_load_default_data_check_shape(self):
        """
        Checks, whether the loaded data has the required shape.
        """
        self.assertEqual(self.data.shape, (3, 2))

    def test_load_default_data_check_columns(self):
        """
        Checks, whether the loaded data has the required columns.
        """
        self.assertEqual(list(self.data.columns), ["text", "lect"])

    def test_load_default_data_check_lects(self):
        """
        Checks, whether the loaded data has the expected set of lects in it.
        """
        expected_lects = ["Slovenian", "Croatian", "Slovak"]

        result_lects = list(self.data["lect"].unique())

        self.assertEqual(expected_lects, result_lects)


class TestLoadData(unittest.TestCase):
    """
    Tests the call of load_data.
    """

    def setUp(self):
        """
        Makes a directory, necessary to emulate the directory that stores the data.
        """
        self.dir = "current_dir"
        mkdir(self.dir)

    def tearDown(self):
        """
        Removes the created directory, in order not to create clutter.
        """
        rmtree(self.dir)
        del self.dir

    def test_load_data_negative_cases_content_dir(self):
        """
        Checks, whether the function throws an error in case content_directory is incorrect.
        """
        negative_cases = ['', ' ', "not_dir", 0, 1, 0.5, [], [""], {}, {"1": 1}, {1: "1"}, {1: 1}]

        test_cases = [
            {
                "content_directory": case,
                "split": 1
            } for case in negative_cases]

        for case in test_cases:
            with self.subTest(data=case):
                self.assertRaises(
                    ValueError,
                    dl.load_data,
                    case["content_directory"],
                    case["split"]
                    )

    def test_load_data_negative_cases_split(self):
        """
        Checks, whether the function throws an error in case split is incorrect.
        """
        negative_cases = ['', ' ', -1, 0, 2, 2.7, [], [""], {}, {"1": 1}, {1: "1"}, {1: 1}]

        test_cases = [
            {
                "content_directory": self.dir,
                "split": case,
            } for case in negative_cases]

        for case in test_cases:
            with self.subTest(data=case):
                self.assertRaises(
                    ValueError,
                    dl.load_data,
                    case["content_directory"],
                    case["split"]
                    )

    def test_load_data_no_files(self):
        """
        Covers the case, when there are no files in the folder.
        """

        warning = 'WARNING:corpus_distance.data_preprocessing.data_loading:'\
            'No data loaded, proceed with caution'

        with self.assertLogs(dl.logger.name, "WARNING") as captured:
            dl.load_data (self.dir, 1)

        self.assertTrue(warning in captured.output)

    def test_load_data_wrong_txt_file(self):
        """
        Assures the correct execution of the function in the base case,
        with split being 1.
        """
        lect = 'dummy'
        text = 'token dummy token'
        with open(join(self.dir, lect + '.txt'), 'w', encoding='utf-8') as f:
            f.write(text)

        self.assertRaises(ValueError, dl.load_data, self.dir)


    def test_load_data_other_file_types(self):
        """
        Covers the case, in which there are no .txt files in the folder.
        """

        with open(join(self.dir, 'dummy.csv'), 'w', encoding='utf-8') as f:
            f.write('a;b;')
        warning = 'WARNING:corpus_distance.data_preprocessing.data_loading:'\
            'No data loaded, proceed with caution'

        with self.assertLogs(dl.logger.name, "WARNING") as captured:
            dl.load_data (self.dir, 1)

        self.assertTrue(warning in captured.output)

    def test_load_data_correct_split_base(self):
        """
        Assures the correct execution of the function in the base case,
        with split being 1.
        """
        lect = 'dummy'
        text = 'token dummy token'
        with open(join(self.dir, 'dummy.' + lect + '.txt'), 'w', encoding='utf-8') as f:
            f.write(text)

        data = dl.load_data (self.dir)
        first_row = list(data.itertuples())[0]

        self.assertEqual(first_row, (0, text, lect))

    def test_load_data_correct_split_changed(self):
        """
        Assures the correct execution of the function in the base case,
        with split being less than 1.
        """
        lect = 'dummy'
        text = 'token dummy token'
        with open(join(self.dir, 'dummy.' + lect + '.txt'), 'w', encoding='utf-8') as f:
            f.write(text)

        data = dl.load_data (self.dir, 0.3)
        first_row = list(data.itertuples())[0]

        self.assertEqual(first_row.text, 'token')

    def test_load_data_mixed_folder(self):
        """
        Assures the correct processing of the folder with mixed types of files.
        """

        lect = 'dummy'
        text = 'token dummy token'
        with open(join(self.dir, 'dummy.' + lect + '.txt'), 'w', encoding='utf-8') as f:
            f.write(text)
        with open(join(self.dir, 'dummy.csv'), 'w', encoding='utf-8') as f:
            f.write('a;b;')

        data = dl.load_data (self.dir, 0.3)

        self.assertEqual(data.shape, (1, 2))


if __name__ == '__main__':
    unittest.main()
