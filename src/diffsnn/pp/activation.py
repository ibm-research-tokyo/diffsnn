#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "Copyright IBM Corp. 2020, 2021"


import torch
from torch import nn
from ..data import MultivariateEventSeq


class SigmoidActivationMixin:

    def init_activation(self, upperbound=1.0, eps=1e-4, **kwargs):
        self.eps = eps
        self._activation_func = nn.Sigmoid()
        self._log_activation_func = nn.LogSigmoid()
        self.upperbound = torch.tensor(upperbound)
        self.activation_func = lambda x: self.upperbound * self._activation_func(x)
        self.log_activation_func = lambda x: (torch.log(self.upperbound) \
                                              + self._log_activation_func(x))

    def upperbound_cond_int(self, history=None, dim=None) -> float:
        ''' the upperbound of the conditional intensity function.
        this is used in the thinning algorithm.
        '''
        if dim is None:
            dim = self.dim
        return (self.upperbound + self.eps) * dim


class ExpActivationMixin:

    def init_activation(self, **kwargs):
        self.activation_func = torch.exp
        self.log_activation_func = lambda x: x

    def upperbound_cond_int(self, history=None, magnitude=4.0) -> float:
        ''' the upperbound of the conditional intensity function.
        this is used in the thinning algorithm.

        for the exponential activation case, this upperbound is heuristic
        '''
        if history is None:
            raise ValueError('history must not be None')
        if history.time_list:
            return magnitude \
                * (self.conditional_intensity(history.time_list[-1], history) \
                   + self.event_impact())
        else:
            return magnitude \
                * (self.conditional_intensity(history.obs_period[1], history) \
                   + self.event_impact())

class HardtanhActivationMixin:

    def init_activation(self, upperbound=1.0, eps=1e-8, **kwargs):
        self._activation_func = nn.Hardtanh(0, upperbound)
        self.upperbound = torch.tensor(upperbound)
        self.eps = eps
        self.activation_func = lambda x: (self._activation_func(x) + eps)
        self.log_activation_func = lambda x: (torch.log(self.activation_func(x)))

    def upperbound_cond_int(self, history=None, dim=None) -> float:
        ''' the upperbound of the conditional intensity function.
        this is used in the thinning algorithm.
        '''
        if dim is None:
            dim = self.dim
        return (self.upperbound + 10.0 * self.eps) * dim
