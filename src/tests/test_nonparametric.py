#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = 'Copyright IBM Corp. 2020, 2021'

import unittest
import torch
from diffsnn.nonparametric.kernel import Kernel
from diffsnn.nonparametric.basis import Epanechnikov

class TestKernel(unittest.TestCase):

    def test_forward(self):
        x_size = 100
        y_size = 50
        x = 5.0 * torch.randn(x_size, 1)
        y = 5.0 * torch.randn(y_size, 1)
        my_kernel = Kernel()
        kernel_val = my_kernel.forward(x, y)
        true_kernel = torch.zeros(x_size, y_size)
        epanechikov = Epanechnikov()
        for each_x_idx in range(x_size):
            for each_y_idx in range(y_size):
                true_kernel[each_x_idx, each_y_idx] = epanechikov(torch.norm(x[each_x_idx] - y[each_y_idx]))
        self.assertTrue(((kernel_val - true_kernel) ** 2).sum() < 1e-5)
