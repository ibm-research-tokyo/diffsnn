#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This implements kernel functions used to approximate the filter function.
'''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = 'Copyright IBM Corp. 2020, 2021'

from bisect import bisect_left, bisect_right
from functools import partial
from typing import List
import torch
from torch.nn import Module, Parameter
import diffsnn.nonparametric.basis


class Kernel(Module):

    ''' kernel class
    '''

    def __init__(self,
                 scale=1.0,
                 p=2.0,
                 basis='Epanechnikov',
                 fix_params=True):
        super().__init__()
        self.init_params(scale, fix_params)
        self.pairwise_distance = partial(torch.cdist, p=p)
        self.basis_func = torch.jit.script(getattr(diffsnn.nonparametric.basis, basis)())

    def init_params(self, scale: float, fix_params: bool):
        self.scale = Parameter(torch.tensor(scale))
        if fix_params: self.fix_params()

    def fix_params(self):
        for each_param in self.parameters():
            each_param.requires_grad = False

    def forward(self, x, y):
        ''' Compute kernel

        Parameters
        ----------
        x : tensor w/ shape (sample_size_1, dim)
        y : tensor w/ shape (sample_size_2, dim)
        '''
        self._check_consistency(x, y)
        return self.basis_func(self.pairwise_distance(x, y) / self.scale)
    '''
        if self.basis_func.support == (-float('inf'), float('inf')) or x.shape[1] != 1:
            return self.basis_func(self.pairwise_distance(x, y) / self.scale)
        else:
            x_org2sort_idx = torch.argsort(x.reshape(-1))
            x_sort2org_idx = torch.argsort(x_org2sort_idx)
            x_sorted_tensor = x[x_org2sort_idx]
            x_sorted_list: List[int] = x_sorted_tensor.reshape(-1).tolist()
            y_org2sort_idx = torch.argsort(y.reshape(-1))
            y_sort2org_idx = torch.argsort(y_org2sort_idx)
            y_sorted_tensor = y[y_org2sort_idx]
            y_sorted_list: List[int] = y_sorted_tensor.reshape(-1).tolist()
            kernel_tensor = torch.zeros(x.shape[0], y.shape[0])

            y_start_idx = 0
            for each_x_idx in range(x.shape[0]):
                y_start_idx = y_start_idx\
                    + bisect_left(y_sorted_list[y_start_idx:],
                                  x_sorted_list[each_x_idx] + self.basis_func.support[0])
                y_end_idx = y_start_idx\
                    + bisect_right(y_sorted_list[y_start_idx:],
                                   x_sorted_list[each_x_idx] + self.basis_func.support[1])
                kernel_tensor[each_x_idx, y_start_idx:y_end_idx+1] \
                    = self.basis_func.forward(
                        self.pairwise_distance(
                            x_sorted_tensor[each_x_idx:each_x_idx+1, :],
                            y_sorted_tensor[y_start_idx:y_end_idx+1, :]) / self.scale)
            return kernel_tensor[x_sort2org_idx, :][:,  y_sort2org_idx]
    ''' 
    @staticmethod
    def _check_consistency(x, y):
        if x.shape[1] != y.shape[1]:
            raise ValueError('x and y have inconsistent shapes')
