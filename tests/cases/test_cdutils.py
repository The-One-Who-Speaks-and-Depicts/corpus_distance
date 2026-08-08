"""
Contains tests of the corpus_distance/cdutils module.
"""
from os import mkdir
from os.path import join
import unittest

from shutil import rmtree
import pandas as pd

from corpus_distance.cdutils import (
    clear_stop_words, return_topic_words,
    get_lects_from_dataframe, get_unique_pairs, LectPair,
    get_to_0_1, delete_outliers,
    create_and_set_storage_directory
)

class TestClearStopWords(unittest.TestCase):
    """
    Checks the behaviour of the clear_stop_words function.
    """

    def test_clear_stop_words_bad_cases_text(self):
        """
        Checks, whether the function correctly processes
        the cases with the wrong argument text.
        """
        self.assertRaises(ValueError, clear_stop_words, 1, ['ab'])
        self.assertRaises(ValueError, clear_stop_words, 0.5, ['ab'])
        self.assertRaises(ValueError, clear_stop_words, False, ['ab'])
        self.assertRaises(ValueError, clear_stop_words, [], ['ab'])
        self.assertRaises(ValueError, clear_stop_words, {}, ['ab'])

        self.assertRaises(ValueError, clear_stop_words, ['ab'], 'ab ba')

    def test_clear_stop_words_bad_cases_stop_words(self):
        """
        Checks, whether the function correctly processes the cases
        with the wrong argument stop_words.
        """
        self.assertRaises(ValueError, clear_stop_words, 'ab ba', 'ab')
        self.assertRaises(ValueError, clear_stop_words, 'ab ba', 1)
        self.assertRaises(ValueError, clear_stop_words, 'ab ba', False)
        self.assertRaises(ValueError, clear_stop_words, 'ab ba', {})
        self.assertRaises(ValueError, clear_stop_words, 'ab ba', 0.5)

    def test_clear_stop_words_void_stop_words(self):
        """
        Checks, whether the function correctly processes the case
        when the stop words list is empty.
        """
        self.assertEqual(clear_stop_words('ab ba'), 'ab ba')
        self.assertEqual(clear_stop_words('ab ba', []), 'ab ba')
        self.assertEqual(clear_stop_words('ab ba', None), 'ab ba')

    def test_clear_stop_words_good_case(self):
        """
        Checks, whether the function correctly handles the good case.
        """
        self.assertEqual(clear_stop_words('ab ba', ['ab']), 'ba')

class TestRemoveTopicWords(unittest.TestCase):
    """
    Checks the behaviour of the clear_stop_words function.
    """

    def test_return_topic_words_bad_cases_text(self):
        """
        Checks, whether the function correctly processes
        the cases with the wrong argument text.
        """
        self.assertRaises(ValueError, return_topic_words, 1, ['ab'])
        self.assertRaises(ValueError, return_topic_words, 0.5, ['ab'])
        self.assertRaises(ValueError, return_topic_words, False, ['ab'])
        self.assertRaises(ValueError, return_topic_words, [], ['ab'])
        self.assertRaises(ValueError, return_topic_words, {}, ['ab'])

        self.assertRaises(ValueError, return_topic_words, ['ab'], 'ab ba')

    def test_return_topic_words_bad_cases_topic_words(self):
        """
        Checks, whether the function correctly processes the cases
        with the wrong argument topic_words.
        """
        self.assertRaises(ValueError, return_topic_words, 'ab ba', 'ab')
        self.assertRaises(ValueError, return_topic_words, 'ab ba', 1)
        self.assertRaises(ValueError, return_topic_words, 'ab ba', False)
        self.assertRaises(ValueError, return_topic_words, 'ab ba', {})
        self.assertRaises(ValueError, return_topic_words, 'ab ba', 0.5)
        self.assertRaises(ValueError, return_topic_words, 'ab ba', [])
        self.assertRaises(ValueError, return_topic_words, 'ab ba', None)


        self.assertRaises(TypeError, return_topic_words, 'ab ba')

    def test_return_topic_words_good_case(self):
        """
        Checks, whether the function correctly handles the good case.
        """
        self.assertEqual(return_topic_words('ab ba', ['ab']), 'ab')

class TestGetLectsFromDataFrame(unittest.TestCase):
    """
    Checks, whether the get_lects_from_data_frame function
    processes the data properly.
    """


    def test_get_lects_from_data_frame_bad_type_cases(self):
        """
        Assures the raising of ValueError in case the user does not
        provide a pandas data frame as an input parameter.
        """
        self.assertRaises(ValueError, get_lects_from_dataframe, 0)
        self.assertRaises(ValueError, get_lects_from_dataframe, 0.5)
        self.assertRaises(ValueError, get_lects_from_dataframe, '0')
        self.assertRaises(ValueError, get_lects_from_dataframe, [])
        self.assertRaises(ValueError, get_lects_from_dataframe, {})

    def test_get_lects_from_data_frame_bad_column_case(self):
        """
        Assures the raising of ValueError in case the user does not
        provide a pandas data frame with the right column in it as an input parameter.
        """
        bad_data_frame = pd.DataFrame.from_dict(
            data={'1': ['lect_1', 'text_1']},
            orient='index',
            columns=['lang', 'text']
            )

        self.assertRaises(ValueError, get_lects_from_dataframe, bad_data_frame)

    def test_get_lects_from_data_frame_good_column_case(self):
        """
        Assures the correct processing of a case with the right column in the data frame.
        """
        good_data_frame = pd.DataFrame.from_dict(
            data={
                '1': ['lect_1', 'text_1'],
                '2': ['lect_2', 'text_2'],
                '3': ['lect_2', 'text_3']
                },
                orient='index',
                columns=['lect', 'text']
                )
        expected = ['lect_1', 'lect_2']

        result = get_lects_from_dataframe(good_data_frame)

        self.assertEqual(expected, result)

class TestGetToNilOne(unittest.TestCase):
    """
    Checks the behaviour of the get_to_0_1 function.
    """

    def test_get_to_0_1_bad_cases(self):
        """
        Checks, whether the function correctly processes the cases
        that contain errors.
        """
        self.assertRaises(ValueError, get_to_0_1, 0)
        self.assertRaises(ValueError, get_to_0_1, {})
        self.assertRaises(ValueError, get_to_0_1, False)
        self.assertRaises(ValueError, get_to_0_1, 0.5)
        self.assertRaises(ValueError, get_to_0_1, 'a')
        self.assertRaises(ValueError, get_to_0_1, '')
        self.assertRaises(ValueError, get_to_0_1, [1, False])


    def test_get_to_0_1_empty(self):
        """
        Checks, whether the function returns empty array in case it received one.
        """
        self.assertEqual([], get_to_0_1([]))


    def test_get_to_0_1_normalised(self):
        """
        Checks, whether the function correctly behaves in case
        when the distribution is already in [0; 1].
        """

        initial_distribution = [0.1, 0.9, 0.9, 0.9]
        expected = initial_distribution

        result = get_to_0_1(initial_distribution)

        self.assertEqual(result, expected)


    def test_get_to_0_1_static_correct(self):
        """
        Checks the behaviour of the function when the input parameter is an array that consists
        of a sequence of the same number that is less or equal to 1.
        """
        initial_distribution = [1, 1, 1, 1]
        expected = [1, 1, 1, 1]

        result = get_to_0_1(initial_distribution)

        self.assertEqual(result, expected)

    def test_get_to_0_1_static_more_than_1(self):
        """
        Checks the behaviour of the function with the input being a list that consists
        of a sequence of the same number that is bigger than 1.
        """
        initial_distribution = [2, 2, 2, 2]
        expected = [1, 1, 1, 1]

        result = get_to_0_1(initial_distribution)

        self.assertEqual(result, expected)

    def test_get_to_0_1_non_static(self):
        """
        Checks the behaviour of the function with the correctly
        formed list.
        """
        initial_distribution = [2, 4, 6, 8]
        expected = [0.25, 0.4166666666666667, 0.5833333333333334, 0.75]

        result = get_to_0_1(initial_distribution)

        self.assertEqual(result, expected)

class TestDeleteOutliers(unittest.TestCase):
    """
    Checks the behaviour of the delete_outliers function.
    """


    def test_delete_outliers_bad_cases(self):
        """
        Checks, whether the function correctly processes
        cases that are clearly erroneous.
        """
        self.assertRaises(ValueError, delete_outliers, 0)
        self.assertRaises(ValueError, delete_outliers, {})
        self.assertRaises(ValueError, delete_outliers, False)
        self.assertRaises(ValueError, delete_outliers, 0.5)
        self.assertRaises(ValueError, delete_outliers, 'a')
        self.assertRaises(ValueError, delete_outliers, '')
        self.assertRaises(ValueError, delete_outliers, [])
        self.assertRaises(ValueError, delete_outliers, [1, False])

    def test_delete_outliers_not_normal_distribution(self):
        """
        Checks, how the function treats non-normal distribution
        (the distribution that is impossible to normalise by
        the deletion of outliers).
        """
        non_normalised_list = [5, 5, 5, 5, 5, 5, 5]

        result = delete_outliers(non_normalised_list)

        self.assertEqual(non_normalised_list, result)


    def test_delete_outliers_no_outliers(self):
        """
        Assesses the function behaviour in case, when the distribution is normal,
        and there are no outliers.
        """
        non_normalised_list = [2, 2.5, 2.5, 3, 3, 3, 3.5, 3.5, 4]
        expected = [2, 2.5, 2.5, 3, 3, 3, 3.5, 3.5, 4]

        result = delete_outliers(non_normalised_list)

        self.assertEqual(expected, result)


    def test_delete_outliers_good_case(self):
        """
        Assesses the behaviour of the function in the ideal case.
        """
        non_normalised_list = [-1000, 2, 2.5, 2.5, 3, 3, 3, 3.5, 3.5, 4, 100]
        expected = [2, 2.5, 2.5, 3, 3, 3, 3.5, 3.5, 4]

        result = delete_outliers(non_normalised_list)

        self.assertEqual(expected, result)

class TestGetUniquePairs(unittest.TestCase):
    """
    Contains tests for the get_unique_pairs function.
    """

    def test_get_unique_pairs_bad_cases(self):
        """
        Checks the processing of incorrect parameters for the function.
        """

        self.assertRaises(ValueError, get_unique_pairs, 0)
        self.assertRaises(ValueError, get_unique_pairs, 0.5)
        self.assertRaises(ValueError, get_unique_pairs, '')
        self.assertRaises(ValueError, get_unique_pairs, {})
        self.assertRaises(ValueError, get_unique_pairs, {'a': 'b'})
        self.assertRaises(ValueError, get_unique_pairs, [])
        self.assertRaises(ValueError, get_unique_pairs, [1])
        self.assertRaises(ValueError, get_unique_pairs, [0.5])
        self.assertRaises(ValueError, get_unique_pairs, [[]])
        self.assertRaises(ValueError, get_unique_pairs, [['a']])
        self.assertRaises(ValueError, get_unique_pairs, [{}])
        self.assertRaises(ValueError, get_unique_pairs, ['a'])

    def test_get_unique_pairs_good_case_single_pair(self):
        """
        Assures the function correctly processes the case of two lects as its input.
        """
        lects = ['a', 'b']
        expected = [LectPair(lect1='a', lect2='b')]

        result = get_unique_pairs(lects)

        self.assertEqual(expected, result)

    def test_get_unique_pairs_good_case_five_lects(self):
        """
        Assures the function correctly processes the case of multiple lects as its input.
        """
        lects = ['a', 'd', 'c', 'x', 'f']
        expected = [
            LectPair(lect1='a', lect2='d'),
            LectPair(lect1='a', lect2='c'),
            LectPair(lect1='a', lect2='x'),
            LectPair(lect1='a', lect2='f'),
            LectPair(lect1='d', lect2='c'),
            LectPair(lect1='d', lect2='x'),
            LectPair(lect1='d', lect2='f'),
            LectPair(lect1='c', lect2='x'),
            LectPair(lect1='c', lect2='f'),
            LectPair(lect1='x', lect2='f'),
        ]

        result = get_unique_pairs(lects)

        self.assertEqual(expected, result)

class TestCreateAndSetStorageDirectory(unittest.TestCase):
    """
    Contains tests for the create_and_set_storage_directory function.
    """

    def tearDown(self):
        rmtree("exp_1", ignore_errors=True)

    def test_create_and_set_storage_directory_negative_cases(self):
        test_cases = [
            {
                "dir": case
            } for case in [
                0,
                0.5,
                1,
                {},
                {'f': 1},
                {1: 'f'},
                {'f': 'f'},
                [],
                ['gfg'],
                '',
                ' '
                ]
                ]

        for case in test_cases:
            with self.subTest(data=case):
                self.assertRaises(
                    ValueError,
                    create_and_set_storage_directory,
                    case["dir"]
                    )
    
    def test_create_and_set_storage_directory_existing_dir_with_files(self):
        """
        Assures the function throws a warning in case the provided directory
        exists and contains some files.
        """
        dir = "exp_1"
        mkdir(dir)
        with open(join(dir, "dummy.txt"), 'w', encoding='utf-8') as file_input:
            file_input.write('dummy')

        self.assertWarns(UserWarning, create_and_set_storage_directory, dir)

    
    def test_create_and_set_storage_directory_not_existing_dir(self):
        """
        Assures the function works correctly, if the directory had not existed before its call.
        """
        dir = "exp_1"

        result = create_and_set_storage_directory(dir)

        self.assertEqual(result, dir)

if __name__ == '__main__':
    unittest.main()
