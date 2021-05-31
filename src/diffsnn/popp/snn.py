#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = 'Copyright IBM Corp. 2020, 2021'

from copy import deepcopy
from .base import POMultivariatePointProcess
from .thinning import MultivariateThinningAlgorithmForPOMixin
from ..pp.snn import SNNBase
from ..pp.activation import (SigmoidActivationMixin,
                             ExpActivationMixin,
                             HardtanhActivationMixin)


class POSNNBase(SNNBase, POMultivariatePointProcess):

    def __init__(self,
                 n_obs_neurons,
                 n_hidden_neurons,
                 connection_tensor=None,
                 n_inducing_points=5,
                 low=0.0,
                 high=10.0,
                 kernel_kwargs={},
                 activation_kwargs={},
                 lmbd=1e-1,
                 seed=43,
                 **kwargs):
        super().__init__(n_neurons=n_obs_neurons+n_hidden_neurons,
                         connection_tensor=connection_tensor,
                         n_inducing_points=n_inducing_points,
                         low=low,
                         high=high,
                         kernel_kwargs=kernel_kwargs,
                         activation_kwargs=activation_kwargs,
                         lmbd=lmbd,
                         seed=seed,
                         **kwargs)
        self.n_obs_neurons = n_obs_neurons
        self.n_hidden_neurons = n_hidden_neurons
        self.obs_neuron_list = list(range(n_obs_neurons))
        self.hidden_neuron_list = list(range(n_obs_neurons,
                                             n_obs_neurons+n_hidden_neurons))

    @property
    def obs_dim(self):
        return self.n_obs_neurons

    @property
    def hidden_dim(self):
        return self.n_hidden_neurons

    def transfer_model(self, train_model):
        self.n_obs_neurons = train_model.n_obs_neurons
        self.n_hidden_neurons = train_model.n_hidden_neurons
        self.n_neurons = train_model.n_neurons
        self.n_inducing_points = train_model.n_inducing_points
        self.connection_tensor = train_model.connection_tensor
        for each_param in self.params:
            self.params[each_param].data = deepcopy(train_model.params[each_param].data)


# reinforce estimator

class SigmoidPOSNN(MultivariateThinningAlgorithmForPOMixin,
                   SigmoidActivationMixin,
                   POSNNBase):
    pass


class ExpPOSNN(MultivariateThinningAlgorithmForPOMixin,
               ExpActivationMixin,
               POSNNBase):
    pass


class HardtanhPOSNN(MultivariateThinningAlgorithmForPOMixin,
                    HardtanhActivationMixin,
                    POSNNBase):
    pass
