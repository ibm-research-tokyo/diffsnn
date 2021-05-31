#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = 'Copyright IBM Corp. 2020, 2021'

import unittest
import numpy as np
import torch
from diffsnn.pp.poisson import PoissonProcess, MultivariatePoissonProcess
from diffsnn.pp.hawkes import (HawkesProcess,
                               NonparametricHawkesProcess,
                               NonparametricHawkesProcessWithStochasticApproximation)
from diffsnn.data import EventSeq

class TestPoissonProcessClass(unittest.TestCase):

    def test_simulation(self):
        ''' check whether `simulate` works correctly for the Poisson process
        '''
        intensity = 2.0
        pp = PoissonProcess(intensity=intensity)
        history_list = pp.simulate(1, [0, 1000])
        time_list = [0] + history_list[0].time_list
        diff_list = [time_list[each_idx+1] - time_list[each_idx] \
                     for each_idx in range(len(time_list) - 1)]
        self.assertTrue(np.abs(intensity - 1.0/np.mean(diff_list)) < 0.2)

    def test_fit(self):
        ''' check whether `fit` works correctly for the Poisson process
        '''
        intensity = np.random.uniform(1e-2, 2.0)
        pp = PoissonProcess(intensity=intensity)
        history_list = pp.simulate(10, [0, 100])
        pp.randomize_params(upper=2.0, lower=1e-2)
        pp.fit(history_list, n_epochs=1000, print_freq=100, optimizer_kwargs={'lr': 1e-1})
        print('- true intensity: {}\n- estimated: {}'.format(intensity, pp.params['intensity']))
        self.assertTrue(torch.abs(intensity - pp.params['intensity']) / intensity < 0.1)

class TestMultivariatePoissonProcessClass(unittest.TestCase):

    def test_simulation(self):
        ''' check whether `simulate` works correctly for the Poisson process
        '''
        intensity_list = [1.0, 0.5]
        pp = MultivariatePoissonProcess(intensity_list=intensity_list)
        history_list = pp.simulate(1, [0, 1000])
        pp.draw_conditional_intensity(history_list[0], 'pp.png')
        time_list = [0] + history_list[0].time_list
        diff_list = [time_list[each_idx+1] - time_list[each_idx] \
                     for each_idx in range(len(time_list) - 1)]
        self.assertTrue(np.abs(sum(intensity_list) - 1.0/np.mean(diff_list)) < 0.2)

    def test_fit(self):
        ''' check whether `fit` works correctly for the Poisson process
        '''
        intensity_list = [np.random.uniform(1e-2, 10) for _ in range(5)]
        pp = MultivariatePoissonProcess(intensity_list=intensity_list)
        history_list = pp.simulate(10, [0, 100])
        pp.randomize_params(upper=10.0, lower=1e-2)
        pp.fit(history_list, n_epochs=500, print_freq=100, optimizer_kwargs={'lr': 1e-1})
        print('- true intensity: {}\n- estimated: {}'\
              .format(intensity_list, pp.params['intensity_list']))
        self.assertTrue(
            torch.abs(torch.tensor(intensity_list) - pp.params['intensity_list']).sum() \
            / torch.tensor(intensity_list).sum() < 0.1)


class TestHawkesProcessClass(unittest.TestCase):

    def setUp(self):
        self.history = EventSeq([0., 1., 2., 3.], [0., 4.])

    def test_conditional_intensity(self):
        ''' check the correctness of conditional intensity function
        '''
        bg_intensity = 1.0
        weight = 1.0
        decay = 1.0
        hawkes = HawkesProcess(bg_intensity=bg_intensity,
                               weight=weight,
                               decay=decay)
        time_stamp = 5.0
        cond_int = bg_intensity
        for each_time_stamp in self.history.time_list:
            cond_int += weight * np.exp(-weight * (time_stamp - each_time_stamp))
        self.assertTrue(np.allclose(
            cond_int,
            hawkes.conditional_intensity(5.0, self.history).item()))

    def test_neg_ll(self):
        bg_intensity = 0.2
        weight = 0.4
        decay = 1.0
        true_hawkes = HawkesProcess(bg_intensity=bg_intensity,
                                    weight=weight,
                                    decay=decay)
        history_list = true_hawkes.simulate(50, [0, 50])
        import time
        t1 = time.time()
        neg_ll = sum([true_hawkes.neg_ll(each_history) for each_history in history_list])
        t2 = time.time()
        neg_ll_wo_recursion = sum([true_hawkes.neg_ll_wo_recursion(each_history) for each_history in history_list])
        t3 = time.time()
        print('neg_ll: {} sec, neg_ll_wo_rec: {} sec'.format(t2 - t1, t3 - t2))
        self.assertTrue(torch.allclose(neg_ll, neg_ll_wo_recursion))

    def test_fit(self):
        bg_intensity = 0.2
        weight = 0.4
        decay = 1.0
        true_hawkes = HawkesProcess(bg_intensity=bg_intensity,
                                    weight=weight,
                                    decay=decay)
        history_list = true_hawkes.simulate(50, [0, 50])
        hawkes = HawkesProcess(bg_intensity=bg_intensity,
                               weight=weight,
                               decay=decay)
        hawkes.randomize_params(3.0, 1e-1, except_for=['bg_intensity',
                                                       'weight',
                                                       'decay'][0:0])
        print(' * initial hawkes\n', hawkes)
        hawkes.fit(history_list,
                   n_epochs=500,
                   print_freq=100, optimizer_kwargs={'lr': 5e-2})
        print(' * true hawkes\n', true_hawkes)
        print(' * learned hawkes\n', hawkes)
        #true_hawkes.draw_conditional_intensity(history_list[1], 'test_truth.png')
        #hawkes.draw_conditional_intensity(history_list[1], 'test_learned.png')

class TestNPHawkesProcessClass(unittest.TestCase):

    def setUp(self):
        self.history = EventSeq([0., 1., 2., 3.], [0., 4.])

    def test_conditional_intensity(self):
        ''' check the correctness of conditional intensity function
        '''
        bg_intensity = 1.0
        hawkes = NonparametricHawkesProcess(bg_intensity=bg_intensity,
                                            n_inducing_points=5)
        hawkes.randomize_params(3.0, 1e-2, except_for=['ind_points'])
        self.assertTrue(torch.allclose(hawkes.conditional_intensity(5.0, self.history),
                                       hawkes._slow_conditional_intensity(5.0, self.history)))

    def test_neg_ll(self):
        ''' check whether neg_ll computation is correctly implemented.
        if RuntimeError is raised, the code is buggy.
        '''
        bg_intensity = 1.0
        hawkes = NonparametricHawkesProcess(bg_intensity=bg_intensity,
                                            n_inducing_points=5)
        hawkes.randomize_params(3.0, 1e-2, except_for=['ind_points'])
        print(hawkes.neg_ll(self.history, debug=True))

    def test_fit(self):
        bg_intensity = 0.2
        true_hawkes = NonparametricHawkesProcess(bg_intensity=bg_intensity,
                                                 n_inducing_points=5)
        true_hawkes.params['kernel_weight'].data = torch.tensor([1., 0., 0., 0., 0.])
        history_list = true_hawkes.simulate(50, [0, 20])
        np_hawkes = NonparametricHawkesProcess(bg_intensity=bg_intensity,
                                               n_inducing_points=5)
        np_hawkes.randomize_params(3.0, 1e-2, except_for=['ind_points'])
        print(' * initial hawkes\n', np_hawkes)
        np_hawkes.fit(history_list,
                      n_epochs=50,
                      print_freq=10, optimizer_kwargs={'lr': 5e-2})
        print(' * true hawkes\n', true_hawkes)
        print(' * learned hawkes\n', np_hawkes)
        #np_hawkes.draw_conditional_intensity(history_list[0], 'np_learned.png')
        #true_hawkes.draw_conditional_intensity(history_list[0], 'np_true.png')

class TestStochasticNPHawkesProcessClass(unittest.TestCase):

    def setUp(self):
        self.history = EventSeq([0., 1., 2., 3.], [0., 4.])

    def test_neg_ll(self):
        ''' check whether neg_ll computation is correctly implemented.
        if RuntimeError is raised, the code is buggy.
        '''
        bg_intensity = 1.0
        hawkes = NonparametricHawkesProcessWithStochasticApproximation(
            bg_intensity=bg_intensity,
            n_inducing_points=5)
        hawkes.randomize_params(3.0, 1e-2, except_for=['ind_points'])
        print(hawkes.neg_ll(self.history, debug=True))

    def test_fit(self):
        bg_intensity = 0.2
        true_hawkes = NonparametricHawkesProcess(bg_intensity=bg_intensity,
                                                 n_inducing_points=5)
        true_hawkes.params['kernel_weight'].data = torch.tensor([1., 0., 0., 0., 0.])
        history_list = true_hawkes.simulate(50, [0, 20])
        np_hawkes = NonparametricHawkesProcessWithStochasticApproximation(
            bg_intensity=bg_intensity,
            n_inducing_points=5)
        np_hawkes.randomize_params(3.0, 1e-2, except_for=['ind_points'])
        print(' * initial hawkes\n', np_hawkes)
        np_hawkes.fit(history_list,
                      n_epochs=50,
                      batch_size=1,
                      print_freq=10, optimizer_kwargs={'lr': 5e-2})
        print(' * true hawkes\n', true_hawkes)
        print(' * learned hawkes\n', np_hawkes)
        #np_hawkes.draw_conditional_intensity(history_list[0], 'np_learned.png')
        #true_hawkes.draw_conditional_intensity(history_list[0], 'np_true.png')
