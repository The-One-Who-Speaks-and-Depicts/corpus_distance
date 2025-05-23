"""
Tests for topic modelling module.
"""

import unittest

from corpus_distance.data_preprocessing import topic_modelling as tm

class TestSetTopicRange(unittest.TestCase):
    """
    Tests function set_topic_range.
    """

    def test_set_topic_range_bad_cases(self):
        """
        Checks bad cases: negative numbers for topic range (both start and finish),
        as well as user-given range of topics to take getting out of bounds of
        the generated topics range. 
        """
        wrong_params = tm.LDAParams()

        wrong_params.required_topics_num = 15

        self.assertRaises(ValueError, tm.set_topic_range, wrong_params)

        wrong_params.required_topics_num = -1

        self.assertRaises(ValueError, tm.set_topic_range, wrong_params)

        wrong_params.required_topics_num = 5
        wrong_params.required_topics_start = 7

        self.assertRaises(ValueError, tm.set_topic_range, wrong_params)

        wrong_params.required_topics_start = -1

        self.assertRaises(ValueError, tm.set_topic_range, wrong_params)


    def test_set_topic_range_good_cases_custom_number(self):
        """
        Covers case of restricted from the top topic range, in which the user
        wants to take the number of topics, which is less than the number of 
        topics, generated by the model, starting from the first (index 0) topic.
        """
        params = tm.LDAParams(required_topics_num=5)

        first_topic, last_topic = tm.set_topic_range(params)

        self.assertEqual(first_topic, 0)
        self.assertEqual(last_topic, 5)

    def test_set_topic_range_good_cases_custom_start(self):
        """
        Covers case of restricted from the top topic range, in which the user
        wants to take all the topics, generated by the model, 
        starting from the topic, index of which user provides.
        """
        params = tm.LDAParams(required_topics_start=2)

        first_topic, last_topic = tm.set_topic_range(params)

        self.assertEqual(first_topic, 2)
        self.assertEqual(last_topic, 10)

    def test_set_topic_range_good_cases_custom_number_and_start(self):
        """
        Covers case of restricted from the top topic range, in which the user
        wants to take the number of topics, which is less than the number of 
        topics, generated by the model, starting from the topic, index of which user provides.
        """
        params = tm.LDAParams(required_topics_start=5, required_topics_num=5)

        first_topic, last_topic = tm.set_topic_range(params)

        self.assertEqual(first_topic, 5)
        self.assertEqual(last_topic, 10)



    def test_set_topic_range_default_case(self):
        """
        Covers the default case, without any user modifications of the first topic to
        take and the range of topics to take.
        """
        first_topic, last_topic = tm.set_topic_range()

        self.assertEqual(first_topic, 0)
        self.assertEqual(last_topic, 10)



if __name__ == '__main__':
    unittest.main()
