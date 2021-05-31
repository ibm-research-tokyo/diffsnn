#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = 'Copyright IBM Corp. 2020, 2021'

from copy import deepcopy
import unittest
import torch
from diffsnn.pp.snn import FullyObsSigmoidSNN
from .base import (TestBase,
                   GenFullyObsSigmoidSNN,
                   GenFullyObsSigmoidSNNWithoutKernel,
                   GenFullyObsSigmoidSNNWithSmallKernel,
                   TrainPoisson,
                   TrainSNN)

# ---------------- main tests----------------


class TestPoissonFit(TrainPoisson,
                     GenFullyObsSigmoidSNNWithoutKernel,
                     TestBase,
                     unittest.TestCase):

    ''' fit Poisson process to SNN w/o kernel weights
    '''
    def setUp(self):
        self.n_neurons = 2
        self.n_epochs = 100
        self.sample_size = 100
        self.length = 50

    def check_fit(self):
        print(' * true conditional_intensity = {}'\
              .format(torch.sigmoid(self.gen_model.params['bias'])))
        print(' * learned one\n\t', self.trainable_model)
        self.assertTrue(torch.allclose(
            self.trainable_model.params['intensity_list'],
            torch.sigmoid(self.gen_model.params['bias']),
            atol=1e-1))

class TestPoissonFitWithNegligibleKernel(GenFullyObsSigmoidSNNWithSmallKernel, TestPoissonFit):
    ''' fit Poisson process to SNN w/ negligible kernel weights
    '''

class TestSNNFit(TrainSNN,
                 GenFullyObsSigmoidSNN,
                 TestBase,
                 unittest.TestCase):

    ''' fit SNN to SNN
    '''

    def setUp(self):
        self.n_neurons = 2
        self.n_epochs = 20
        self.sample_size = 500
        self.length = 50

    def check_fit(self):
        print(' * true snn\n', self.gen_model)
        print(' * learned snn\n', self.trainable_model)
        self.assertTrue(torch.allclose(self.gen_model.params['bias'],
                                       self.trainable_model.params['bias'],
                                       atol=2e-1))

class TestSNNBiasFit(TrainSNN,
                     GenFullyObsSigmoidSNN,
                     TestBase,
                     unittest.TestCase):

    ''' fit SNN bias to SNN
    '''

    def setUp(self):
        self.n_neurons = 2
        self.n_epochs = 10
        self.sample_size = 500
        self.length = 50

    def preprocess(self):
        self.trainable_model.params['kernel_weight'].data \
            = deepcopy(self.gen_model.params['kernel_weight'].data)
        self.trainable_model.params['kernel_weight'].requires_grad = False

    def check_fit(self):
        print(' * true snn\n', self.gen_model)
        print(' * learned snn\n', self.trainable_model)
        self.assertTrue(torch.allclose(self.gen_model.params['bias'],
                                       self.trainable_model.params['bias'],
                                       atol=2e-1))
