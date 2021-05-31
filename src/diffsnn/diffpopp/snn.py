#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = 'Copyright IBM Corp. 2020, 2021'

import torch
from .base import (MultivariateDifferentiablePointProcess,
                   HeuristicMultivariateDifferentiablePointProcess)
from .thinning import MultivariateDiffThinningAlgorithmMixin
from ..utils import complete_logprob
from ..popp.snn import POSNNBase
from ..pp.activation import (SigmoidActivationMixin,
                             ExpActivationMixin,
                             HardtanhActivationMixin)


class DiffSNNBase(POSNNBase, MultivariateDifferentiablePointProcess):

    def __init__(self,
                 n_obs_neurons,
                 n_hidden_neurons,
                 connection_tensor=None,
                 n_inducing_points=5,
                 low=0.0,
                 high=10.0,
                 kernel_kwargs={},
                 activation_kwargs={},
                 temperature=1.0,
                 lmbd=1e-1,
                 hard=False,
                 seed=43):
        super().__init__(n_obs_neurons,
                         n_hidden_neurons,
                         connection_tensor,
                         n_inducing_points,
                         low,
                         high,
                         kernel_kwargs,
                         activation_kwargs,
                         lmbd,
                         seed)
        self.temperature = torch.tensor(temperature)
        self.hard = hard

    def conditional_intensity(self,
                              time,
                              log_mark,
                              history):
        cond_int = 0
        # for an observable spike
        if 0 in log_mark[:self.obs_dim]:
            cond_int = self.activation_func(
                self.membrane_potential(
                    time,
                    history,
                    dim_list=[torch.where(log_mark == 0)[0].item()]))
        else: # for a hidden spike
            cond_int = torch.exp(self.log_conditional_intensity(time, log_mark, history))
        if torch.isnan(cond_int):
            raise ValueError(
                'encounter nan in computing conditional intensity')
        return cond_int

    def log_conditional_intensity(self,
                                  time,
                                  log_mark,
                                  history):
        log_cond_int = 0
        if 0 in log_mark[:self.obs_dim]:
            log_cond_int = self.log_activation_func(
                self.membrane_potential(
                    time,
                    history,
                    dim_list=[torch.where(log_mark == 0)[0].item()]))
        else:
            logits = self.log_activation_func(
                self.membrane_potential(
                    time,
                    history,
                    dim_list=list(range(self.obs_dim,
                                        self.obs_dim+self.hidden_dim))))\
                    - torch.log(self.upperbound_cond_int(history,
                                                         dim=self.hidden_dim))
            logits = complete_logprob(logits)
            log_hidden_mark = log_mark[self.obs_dim:] # (hidden_dim + 1)-dim array
            log_cond_int = torch.log(self.upperbound_cond_int(history,
                                                              dim=self.hidden_dim)) \
                + self.expconcrete_dist.log_pdf(
                    log_hidden_mark,
                    self.temperature,
                    logits)
        if torch.isnan(log_cond_int):
            raise ValueError(
                'encounter nan in computing log conditional intensity')
        return log_cond_int


class HeuristicDiffSNNBase(
        POSNNBase,
        HeuristicMultivariateDifferentiablePointProcess):

    def __init__(self,
                 n_obs_neurons,
                 n_hidden_neurons,
                 connection_tensor=None,
                 n_inducing_points=5,
                 low=0.0,
                 high=10.0,
                 kernel_kwargs={},
                 activation_kwargs={},
                 temperature=1.0,
                 lmbd=1e-1,
                 hard=False,
                 seed=43):
        super().__init__(n_obs_neurons,
                         n_hidden_neurons,
                         connection_tensor,
                         n_inducing_points,
                         low,
                         high,
                         kernel_kwargs,
                         activation_kwargs,
                         lmbd,
                         seed)
        self.temperature = torch.tensor(temperature)
        self.hard = hard

    def conditional_intensity(self,
                              time,
                              log_mark,
                              history):
        return torch.exp(self.log_conditional_intensity(time, log_mark, history))

    def log_conditional_intensity(self,
                                  time,
                                  log_mark,
                                  history):
        mark = torch.exp(log_mark[:-1])
        if mark.sum() == 0:
            return 0
        else:
            return self.log_activation_func(
                self.membrane_potential(time, history)) @ mark


class SigmoidDiffSNN(MultivariateDiffThinningAlgorithmMixin,
                     SigmoidActivationMixin,
                     DiffSNNBase):
    pass


class ExpDiffSNN(MultivariateDiffThinningAlgorithmMixin,
                 ExpActivationMixin,
                 DiffSNNBase):
    pass


class HardtanhDiffSNN(MultivariateDiffThinningAlgorithmMixin,
                      HardtanhActivationMixin,
                      DiffSNNBase):
    pass

class HeuristicSigmoidDiffSNN(MultivariateDiffThinningAlgorithmMixin,
                              SigmoidActivationMixin,
                              HeuristicDiffSNNBase):
    pass


class HeuristicExpDiffSNN(MultivariateDiffThinningAlgorithmMixin,
                          ExpActivationMixin,
                          HeuristicDiffSNNBase):
    pass


class HeuristicHardtanhDiffSNN(MultivariateDiffThinningAlgorithmMixin,
                               HardtanhActivationMixin,
                               HeuristicDiffSNNBase):
    pass
