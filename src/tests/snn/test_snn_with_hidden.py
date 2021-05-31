#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = 'Copyright IBM Corp. 2020, 2021'

from copy import deepcopy
import unittest
import torch
from diffsnn.pp.poisson import MultivariatePoissonProcess
from diffsnn.data import MultivariateDiffEventSeq
from .base import (DiffTestBase,
                   POTestBase,
                   GenDiffSigmoidSNN,
                   GenDiffSigmoidSNNWithoutKernel,
                   GenDiffHardtanhSNN,
                   GenHardtanhPOSNN,
                   TrainObsPoisson,
                   TrainPOSNN,
                   TrainDiffPOSNN)


class TestPoissonFit(TrainObsPoisson,
                     GenDiffSigmoidSNNWithoutKernel,
                     DiffTestBase,
                     unittest.TestCase):

    ''' fit Poisson process to SNN w/o kernel weights
    '''

    def mod_params(self):
        self.n_epochs = 50
        self.sample_size = 100
        self.length = 50

    def check_fit(self):
        print(' * true conditional_intensity = {}'\
              .format(torch.sigmoid(self.gen_model.params['bias'])[:self.n_obs_neurons]))
        print(' * learned one\n\t', self.trainable_model)
        self.assertTrue(torch.allclose(
            self.trainable_model.params['intensity_list'][:self.n_obs_neurons],
            torch.sigmoid(self.gen_model.params['bias'][:self.n_obs_neurons]),
            atol=1e-1))


class TestSNNBiasFit(TrainDiffPOSNN,
                     GenDiffSigmoidSNNWithoutKernel,
                     DiffTestBase,
                     unittest.TestCase):

    ''' fit SNN bias to SNN
    '''

    def mod_params(self):
        self.n_epochs = 10
        self.sample_size = 100
        self.length = 50
        self.obj_func_kwargs = {'n_pos': 50,
                                'n_neg': 50,
                                'n_sampling': 1,
                                'beta': 1.}

    def preprocess(self):
        self.trainable_model.params['kernel_weight'].data \
            = deepcopy(self.gen_model.params['kernel_weight'].data)
        self.trainable_model.params['kernel_weight'].requires_grad = False

    def check_fit(self):
        print(' * true snn\n', self.gen_model)
        print(' * learned snn\n', self.trainable_model)
        self.assertTrue(torch.allclose(self.trainable_model.params['bias'][:self.n_obs_neurons],
                                       self.gen_model.params['bias'][:self.n_obs_neurons],
                                       atol=0.2, rtol=0.5))


class TestSNNBiasFitWithKernel(TrainDiffPOSNN,
                               GenDiffSigmoidSNN,
                               DiffTestBase,
                               unittest.TestCase):

    ''' fit SNN bias to SNN
    '''
    def mod_params(self):
        self.n_epochs = 5
        self.sample_size = 10
        self.length = 50
        self.obj_func_kwargs = {'n_pos': 10,
                                'n_neg': 10,
                                'n_sampling': 1,
                                'beta': 1.}

    def preprocess(self):
        '''
        self.trainable_model.params['kernel_weight'].data \
            = deepcopy(self.gen_model.params['kernel_weight'].data)
        '''
        #self.trainable_model.params['kernel_weight'].requires_grad = False

    def check_fit(self):
        print(' * true snn\n', self.gen_model)
        print(' * learned snn\n', self.trainable_model)
        print(' * var snn\n', self.var_model)
        '''
        self.assertTrue(torch.allclose(self.gen_model.params['bias'][:self.n_obs_neurons],
                                       self.trainable_model.params['bias'][:self.n_obs_neurons],
                                       atol=2e-1))
        '''


class TestHardtanhDiffSNN(TrainDiffPOSNN,
                          GenDiffHardtanhSNN,
                          DiffTestBase,
                          unittest.TestCase):

    ''' fit SNN bias to SNN
    '''
    def mod_params(self):
        self.n_epochs = 5
        self.sample_size = 10
        self.length = 50
        self.temperature = 0.3
        self.temperature_rate = 1.
        self.obj_func_kwargs = {'n_pos': 100,
                                'n_neg': 100,
                                'n_sampling': 1,
                                'beta': 1.}

    def preprocess(self):
        '''
        self.trainable_model.params['kernel_weight'].data \
            = deepcopy(self.gen_model.params['kernel_weight'].data)
        self.trainable_model.params['kernel_weight'].requires_grad = True
        '''
        print(' * initial snn\n', self.trainable_model)

    def check_fit(self):
        print(' * true snn\n', self.gen_model)
        print(' * learned snn\n', self.trainable_model)
        #print(' * var snn\n', self.var_model)
        '''
        self.assertTrue(torch.allclose(self.gen_model.params['bias'][:self.n_obs_neurons],
                                       self.trainable_model.params['bias'][:self.n_obs_neurons],
                                       atol=2e-1))
        '''

class TestHardtanhPOSNN(TrainPOSNN,
                        GenHardtanhPOSNN,
                        POTestBase,
                        unittest.TestCase):

    ''' fit SNN bias to SNN
    '''
    def mod_params(self):
        self.n_epochs = 10
        self.sample_size = 10
        self.length = 50
        self.obj_func_kwargs = {'n_pos': 10,
                                'n_neg': 10,
                                'n_sampling': 1,
                                'use_variational': False}

    def preprocess(self):
        '''
        self.trainable_model.params['kernel_weight'].data \
            = deepcopy(self.gen_model.params['kernel_weight'].data)
        self.trainable_model.params['kernel_weight'].requires_grad = True
        '''
        print(' * initial snn\n', self.trainable_model)

    def check_fit(self):
        print(' * true snn\n', self.gen_model)
        print(' * learned snn\n', self.trainable_model)
        print(' * var snn\n', self.var_model)
        '''
        self.assertTrue(torch.allclose(self.gen_model.params['bias'][:self.n_obs_neurons],
                                       self.trainable_model.params['bias'][:self.n_obs_neurons],
                                       atol=2e-1))
        '''

