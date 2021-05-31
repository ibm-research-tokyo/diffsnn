#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = 'Copyright IBM Corp. 2020, 2021'

from abc import abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from .activation import (SigmoidActivationMixin,
                         ExpActivationMixin,
                         HardtanhActivationMixin)
from .base import MultivariatePointProcess
from .thinning import MultivariateThinningAlgorithmMixin
from ..checker import is_n_d_tensor
from ..data import MultivariateEventSeq
from ..nonparametric.kernel import Kernel


class SNNBase(MultivariatePointProcess):

    '''
    kernel_weight[k,i,j] represents the k-th weight from neuron j to neuron i
    '''

    def __init__(self,
                 n_neurons,
                 connection_tensor=None,
                 n_inducing_points=5,
                 low=0.0,
                 high=10.0,
                 kernel_kwargs={},
                 activation_kwargs={},
                 lmbd=1e-1,
                 seed=43,
                 **kwargs):
        if connection_tensor is None:
            connection_tensor = torch.tensor([[True] * n_neurons] * n_neurons)
        if isinstance(connection_tensor, (list, tuple)):
            connection_tensor = torch.tensor(connection_tensor)
        if connection_tensor.shape != (n_neurons, n_neurons):
            raise ValueError('connection_tensor.shape must be {}'.format(n_neurons, n_neurons))
        if connection_tensor.dtype != torch.bool:
            raise ValueError('connection_tensor must be a bool-valued tensor.')
        super().__init__(seed=seed)
        self.n_neurons = n_neurons
        self.connection_tensor = connection_tensor
        self.n_inducing_points = n_inducing_points
        self.lmbd = lmbd
        self.init_params(low, high, n_inducing_points)
        self.init_activation(**activation_kwargs)
        self.base_kernel = Kernel(**kernel_kwargs)

    def init_params(self, low, high, n_inducing_points):
        self.params = nn.ParameterDict()
        self.params['bias'] = nn.Parameter(
            torch.zeros(self.n_neurons),
            requires_grad=True)
        self.params['kernel_weight'] = nn.Parameter(
            torch.zeros(self.n_inducing_points,
                        self.n_neurons,
                        self.n_neurons),
            requires_grad=True)
        self.params['ind_points'] = nn.Parameter(
            torch.linspace(low,
                           high,
                           n_inducing_points).reshape(n_inducing_points, 1),
            requires_grad=False)

    @abstractmethod
    def init_activation(self, **kwargs):
        raise NotImplementedError

    def event_impact(self) -> torch.FloatTensor:
        return torch.abs(self.connection_tensor * self.params['kernel_weight']).sum()

    @property
    def dim(self):
        return self.n_neurons

    def randomize_params(self, upper=2.0, lower=-2.0, except_for=[]):
        for each_param in self.params:
            if each_param not in except_for and self.params[each_param].requires_grad:
                nn.init.uniform_(self.params[each_param], lower, upper)
        if 'kernel_weight' not in except_for:
            self.params['kernel_weight'].data = self.params['kernel_weight'].data\
                                                * self.connection_tensor
            for each_neuron in range(self.n_neurons):
                self.params['kernel_weight'][:, each_neuron, each_neuron]\
                    .data.clamp_(-float('inf'), -1e-1)

    def randomize_diagonal(self, upper=-0.1, lower=-2.0):
        for each_weight_idx in range(self.n_inducing_points):
            for each_neuron in range(self.n_neurons):
                nn.init.uniform_(
                    self.params['kernel_weight'][each_weight_idx, each_neuron, each_neuron],
                    lower, upper)

    def clamp_params(self):
        for each_neuron in range(self.n_neurons):
            self.params['kernel_weight'][:, each_neuron, each_neuron]\
                .data.clamp_(-float('inf'), -1e-1)

    def batch_kernel(self, td_tensor: torch.tensor, dim_list=[]):
        '''

        Parameters
        ----------
        td_tensor : torch.tensor
            shape (n_spikes, )

        Returns
        -------
        torch.tensor
            shape (n_spikes, n_neurons, n_neurons)
        '''
        if not is_n_d_tensor(td_tensor, 1):
            raise ValueError('td must be 1-d tensor.')
        # base_kernel_vals[l,k]: the k-th kernel value of the l-th event 
        base_kernel_vals = self.base_kernel.forward(
            td_tensor.view(-1, 1).to(torch.float32),
            self.params['ind_points'])
        filter_vals = 0
        if dim_list:
            n_tgt_neurons = len(dim_list)
            # filter_vals[l,i,j] : the impact of l-th past event, from neuron j to neuron i, where i beongs to dim_list
            # filter_vals = torch.einsum('lk,kij->lij', base_kernel_vals, self.params['kernel_weight'][:, dim_list, :])
            filter_vals = (base_kernel_vals \
                           @ ((self.connection_tensor[dim_list, :] * self.params['kernel_weight'][:, dim_list, :])\
                              .view(self.n_inducing_points, -1)))\
                              .view(-1,
                                    n_tgt_neurons,
                                    self.n_neurons)
        else:
            # filter_vals[l,i,j] : the impact of l-th past event, from neuron j to neuron i
            #filter_vals = torch.einsum('lk,kij->lij', base_kernel_vals, self.params['kernel_weight'])
            # the above expression and the below one are equivalent
            filter_vals = (base_kernel_vals \
                           @ ((self.connection_tensor * self.params['kernel_weight'])\
                              .view(self.n_inducing_points, -1)))\
                              .view(-1,
                                    self.n_neurons,
                                    self.n_neurons)
        return filter_vals

    def membrane_potential(self,
                           time: float,
                           history: MultivariateEventSeq,
                           dim_list=[]) -> torch.Tensor:
        time_tensor, mark_tensor = history.batch_time_mark_tensor(
            time,
            self.params['ind_points'][-1].item() + self.base_kernel.basis_func.support[1])
        potential = 0
        if dim_list:
            potential = self.params['bias'][dim_list]
        else:
            potential = self.params['bias']
        if len(time_tensor) != 0:
            # mark_tensor[l,j]: the l-th (past) event occurs at neuron j or not
            # self.batch_kernel(time - time_tensor)[l,i,j]:
            # the impact of l-th past event, from neuron j to neuron i
            potential = potential \
                + (self.batch_kernel(time - time_tensor, dim_list=dim_list)
                   * mark_tensor.unsqueeze(1)).sum(dim=(0,2))
        return potential

    def all_conditional_intensity(self,
                                  time: float,
                                  history: MultivariateEventSeq,
                                  dim_list=[]) -> torch.Tensor:
        return self.activation_func(self.membrane_potential(time, history, dim_list=dim_list))

    def all_log_conditional_intensity(self,
                                      time: float,
                                      history: MultivariateEventSeq,
                                      dim_list=[]) -> torch.Tensor:
        return self.log_activation_func(self.membrane_potential(time, history, dim_list=dim_list))

    def regularize(self):
        reg = 0
        for each_param in self.params:
            if each_param == 'kernel_weight':
                reg = reg + torch.norm(self.connection_tensor * self.params[each_param])
        return self.lmbd * reg

    @abstractmethod
    def upperbound_cond_int(self, history: MultivariateEventSeq) -> float:
        ''' the upperbound of the conditional intensity function.
        this is used in the thinning algorithm.
        '''
        raise NotImplementedError

    def draw_filter(self, file_name: str):
        fig, ax_list = plt.subplots(self.n_neurons,
                                    self.n_neurons,
                                    figsize=(3 * self.n_neurons, 3 * self.n_neurons))
        td_array = np.linspace(0, 10, 1000)
        filter_vals = self.batch_kernel(torch.tensor(td_array)).detach().numpy()
        for each_from_neuron in range(self.n_neurons):
            for each_to_neuron in range(self.n_neurons):
                ax_list[each_from_neuron][each_to_neuron].plot(
                    td_array,
                    filter_vals[:,
                                each_to_neuron,
                                each_from_neuron])
                ax_list[each_from_neuron][each_to_neuron]\
                    .set_title('Filter from neuron {} to neuron {}'\
                               .format(each_from_neuron,
                                       each_to_neuron))
        plt.tight_layout()
        plt.savefig(file_name)
        plt.clf()


class FullyObsSigmoidSNN(MultivariateThinningAlgorithmMixin,
                         SigmoidActivationMixin,
                         SNNBase):
    pass

class FullyObsExpSNN(MultivariateThinningAlgorithmMixin,
                     ExpActivationMixin,
                     SNNBase):
    pass

class FullyObsHardtanhSNN(MultivariateThinningAlgorithmMixin,
                          HardtanhActivationMixin,
                          SNNBase):
    pass
