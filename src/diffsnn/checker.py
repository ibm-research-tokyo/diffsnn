#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = 'Copyright IBM Corp. 2020, 2021'

import torch

def is_2d_tensor(x_tensor):
    return isinstance(x_tensor, torch.Tensor) and len(x_tensor.shape) == 2

def is_n_d_tensor(x_tensor, n_dim):
    return isinstance(x_tensor, torch.Tensor) and len(x_tensor.shape) == n_dim

def is_shape(x_tensor, shape):
    return x_tensor.shape == torch.Size(shape)
