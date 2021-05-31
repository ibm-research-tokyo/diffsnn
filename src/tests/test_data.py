#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = 'Copyright IBM Corp. 2020, 2021'

import unittest
import torch
from diffsnn.data import (EventSeq,
                          MarkedEventSeq,
                          MultivariateEventSeq,
                          MultivariateDiffEventSeq)


class TestDataClass(unittest.TestCase):

    def setUp(self):
        self.n_events = 10
        self.n_mark_types = 5
        self.n_obs_dim = 3
        self.n_hidden_dim = 2
        self.time_list, _ = torch.rand(self.n_events).sort()
        self.time_list = self.time_list.tolist()
        self.mark_list = torch.randint(self.n_mark_types, size=(self.n_events,)) % self.n_obs_dim
        self.onehot_list = list(torch.unbind(
            torch.nn.functional.one_hot(self.mark_list,
                                        num_classes=self.n_mark_types)))
        self.mark_list = self.mark_list.tolist()
        self.obs_period = [0, 1]

    def test_marked_event_seq(self):
        _ = MarkedEventSeq(self.time_list, self.mark_list, self.obs_period)

    def test_multivariate_event_seq(self):
        # incorrect
        with self.assertRaises(ValueError, msg='mark must be one-hot 2d torch tensor.'):
            _ = MultivariateEventSeq(self.time_list, self.mark_list, self.obs_period)

        # correct
        _ = MultivariateEventSeq(self.time_list, self.onehot_list, self.obs_period)

    def test_multivariate_diff_event_seq(self):
        # incorrect
        with self.assertRaises(ValueError, msg='mark must be within a probability simplex.'):
            _ = MultivariateDiffEventSeq(self.time_list,
                                         self.mark_list,
                                         self.n_obs_dim,
                                         self.n_hidden_dim,
                                         self.obs_period)

        # correct
        _ = MultivariateDiffEventSeq(self.time_list,
                                     self.onehot_list,
                                     self.n_obs_dim,
                                     self.n_hidden_dim,
                                     self.obs_period)

    def test_addition_eventseq(self):
        event_seq_1 = EventSeq(time_list=[0., 1.2, 1.4, 1.6, 2.], obs_period=[0., 2.])
        event_seq_2 = EventSeq(time_list=[0.1, 0.2, 0.3, 0.4], obs_period=[0., 2.])
        event_seq = event_seq_1 + event_seq_2
        self.assertEqual(event_seq.time_list, [0., 0.1, 0.2, 0.3, 0.4,
                                               1.2, 1.4, 1.6, 2.])
        self.assertEqual(event_seq.obs_period, [0., 2.])

    def test_addition_marked(self):
        event_seq_1 = MarkedEventSeq(time_list=[0., 1.2, 1.4, 1.6, 2.],
                                     mark_list=[0., 1.2, 1.4, 1.6, 2.],
                                     obs_period=[0., 2.])
        event_seq_2 = MarkedEventSeq(time_list=[0.1, 0.2, 0.3, 0.4],
                                     mark_list=[0.1, 0.2, 0.3, 0.4],
                                     obs_period=[0., 2.])
        event_seq = event_seq_1 + event_seq_2
        self.assertEqual(event_seq.time_list, [0., 0.1, 0.2, 0.3, 0.4,
                                               1.2, 1.4, 1.6, 2.])
        self.assertEqual(event_seq.mark_list, [0., 0.1, 0.2, 0.3, 0.4,
                                               1.2, 1.4, 1.6, 2.])
        self.assertEqual(event_seq.obs_period, [0., 2.])

    def test_insert(self):
        event_seq = MarkedEventSeq(time_list=[0., 1.2, 1.4, 1.6, 2.],
                                   mark_list=[0., 1.2, 1.4, 1.6, 2.],
                                   obs_period=[0., 2.])
        event_seq.insert(0.5, 0.4)
        self.assertEqual(event_seq.time_list, [0., 0.5, 1.2, 1.4, 1.6, 2.])
        self.assertEqual(event_seq.mark_list, [0., 0.4, 1.2, 1.4, 1.6, 2.])
        event_seq.insert(2.1, 0.1)
        self.assertEqual(event_seq.time_list, [0., 0.5, 1.2, 1.4, 1.6, 2., 2.1])
        self.assertEqual(event_seq.mark_list, [0., 0.4, 1.2, 1.4, 1.6, 2., 0.1])
        self.assertEqual(event_seq.obs_period, [0., 2.1])
        event_seq.insert(-2.1, -0.1)
        self.assertEqual(event_seq.time_list, [-2.1, 0., 0.5, 1.2, 1.4, 1.6, 2., 2.1])
        self.assertEqual(event_seq.mark_list, [-0.1, 0., 0.4, 1.2, 1.4, 1.6, 2., 0.1])
        self.assertEqual(event_seq.obs_period, [-2.1, 2.1])

    def test_sample(self):
        time_list = [0., 1.2, 1.4, 1.6, 2.]
        event_seq = EventSeq(time_list=time_list, obs_period=[0., 2.])
        sampled_time_list = event_seq.sample()
        self.assertTrue(set(sampled_time_list).issubset(set(time_list)))

    def test_sample_multivariate(self):
        time_list = [0., 1.2, 1.4, 1.6]
        mark_list = [0.1, 0.2, 0.3, 0.4]
        size = 10
        event_seq = MarkedEventSeq(time_list=time_list,
                                   mark_list=mark_list,
                                   obs_period=[0., 2.])
        sampled_time_mark_list = event_seq.sample(size=size)
        for each_trial in range(size):
            sampled_idx = time_list.index(sampled_time_mark_list[each_trial][0])
            self.assertEqual(mark_list[sampled_idx], sampled_time_mark_list[each_trial][1])
