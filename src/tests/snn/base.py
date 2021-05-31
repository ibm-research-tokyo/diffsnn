#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = 'Copyright IBM Corp. 2020, 2021'

import torch
from diffsnn.pp.poisson import MultivariatePoissonProcess
from diffsnn.pp.snn import FullyObsSigmoidSNN
from diffsnn.popp.snn import HardtanhPOSNN
from diffsnn.diffpopp.snn import HardtanhDiffSNN, SigmoidDiffSNN
from diffsnn.data import delete_hidden


upperbound = 1.0

class TestBase:

    def setUp(self):
        self.n_neurons = 2
        self.n_epochs = 100
        self.sample_size = 100
        self.length = 50
        self.mod_params()

    @property
    def gen_model(self):
        raise NotImplementedError

    @property
    def trainable_model(self):
        raise NotImplementedError

    def check_fit(self):
        raise NotImplementedError

    def preprocess(self):
        pass

    def mod_params(self):
        pass

    def test_fit(self):
        '''
        import cProfile
        import pstats
        pr = cProfile.Profile()
        pr.enable()
        '''
        history_list = self.gen_model.simulate(self.sample_size,
                                               [0, self.length])
        '''
        pr.disable()
        stats = pstats.Stats(pr)
        stats.sort_stats('tottime')
        #stats.print_stats()
        '''
        self.preprocess()
        self.trainable_model.fit(history_list,
                                 n_epochs=self.n_epochs,
                                 print_freq=max(self.n_epochs // 10, 1),
                                 optimizer_kwargs={'lr': 5e-2})
        self.check_fit()

class POTestBase(TestBase):

    def setUp(self):
        self.n_obs_neurons = 2
        self.n_hidden_neurons = 1
        self.n_neurons = self.n_obs_neurons + self.n_hidden_neurons
        self.n_epochs = 10
        self.sample_size = 100
        self.length = 50
        self.connection_tensor = torch.tensor([[True, False, True],
                                               [False, True, False],
                                               [False, True, True]])
        self.obj_func_kwargs = {'n_pos': 50,
                                'n_neg': 50,
                                'gumbel_softmax_kwargs': {'tau': 1.0,
                                                          'hard': False}}
        self.mod_params()

    @property
    def var_model(self):
        return None

    def test_fit(self):
        obs_history_list = self.gen_model.simulate(self.sample_size,
                                                   [0, self.length])
        self.gen_model.draw_conditional_intensity(obs_history_list[0], 'true.png')
        history_list = [delete_hidden(each_history,
                                      self.n_obs_neurons) \
                        for each_history in obs_history_list]
        self.preprocess()
        self.trainable_model.fit(history_list,
                                 variational_dist=self.var_model,
                                 n_epochs=self.n_epochs,
                                 print_freq=max(self.n_epochs // 10, 1),
                                 optimizer_kwargs={'lr': 5e-2},
                                 obj_func_kwargs=self.obj_func_kwargs)
        self.trainable_model.draw_conditional_intensity(obs_history_list[0], 'learned.png')
        self.check_fit()

class DiffTestBase(POTestBase):

    def setUp(self):
        self.n_obs_neurons = 2
        self.n_hidden_neurons = 1
        self.n_neurons = self.n_obs_neurons + self.n_hidden_neurons
        self.n_epochs = 10
        self.sample_size = 10
        self.length = 50
        self.connection_tensor = torch.tensor([[True, False, True],
                                               [False, True, False],
                                               [False, True, True]])
        self.obj_func_kwargs = {'n_pos': 50,
                                'n_neg': 50}
        self.temperature = 0.3
        self.temperature_rate = 0.95
        self.mod_params()

    def test_fit(self):
        obs_history_list = self.gen_model.simulate(self.sample_size,
                                                   [0, self.length])
        self.gen_model.draw_conditional_intensity(obs_history_list[0], 'true.png')
        history_list = [delete_hidden(each_history,
                                      self.n_obs_neurons) \
                        for each_history in obs_history_list]
        self.preprocess()
        self.trainable_model.fit(history_list,
                                 variational_dist=self,
                                 n_epochs=self.n_epochs,
                                 print_freq=max(self.n_epochs // 10, 1),
                                 optimizer_kwargs={'lr': 5e-2},
                                 temperature_rate=self.temperature_rate,
                                 obj_func_kwargs=self.obj_func_kwargs)
        self.trainable_model.draw_conditional_intensity(obs_history_list[0], 'learned.png')
        self.check_fit()

# ---------------- data generation models ----------------

class GenFOBase:

    @property
    def kernel_weight(self):
        raise NotImplementedError

    @property
    def bias(self):
        raise NotImplementedError

    @property
    def model_class(self):
        raise NotImplementedError

    @property
    def gen_model(self):
        if not hasattr(self, '_gen_model'):
            torch.manual_seed(0)
            connection_tensor = torch.tensor([[True] * self.n_neurons] * self.n_neurons)
            self._gen_model = self.model_class(
                n_neurons=self.n_neurons,
                connection_tensor=connection_tensor)
            self._gen_model.params['bias'].data \
                = self.bias
            self._gen_model.params['kernel_weight'].data \
                = self.kernel_weight
        return self._gen_model


class GenPOBase(GenFOBase):

    @property
    def gen_model(self):
        if not hasattr(self, '_gen_model'):
            torch.manual_seed(0)
            self._gen_model = self.model_class(
                n_obs_neurons=self.n_obs_neurons,
                n_hidden_neurons=self.n_hidden_neurons,
                connection_tensor=self.connection_tensor,
                n_inducing_points=self.kernel_weight.shape[0],
                activation_kwargs={'upperbound': upperbound})
            self._gen_model.params['bias'].data = self.bias
            self._gen_model.params['kernel_weight'].data \
                = self.kernel_weight
        return self._gen_model

class GenDiffPOBase(GenFOBase):

    @property
    def gen_model(self):
        if not hasattr(self, '_gen_model'):
            torch.manual_seed(0)
            self._gen_model = self.model_class(
                n_obs_neurons=self.n_obs_neurons,
                n_hidden_neurons=self.n_hidden_neurons,
                connection_tensor=self.connection_tensor,
                n_inducing_points=self.kernel_weight.shape[0],
                activation_kwargs={'upperbound': upperbound},
                temperature=self.temperature)
            self._gen_model.params['bias'].data = self.bias
            self._gen_model.params['kernel_weight'].data \
                = self.kernel_weight
            self._gen_model.hard = True
        return self._gen_model


class GenFullyObsSigmoidSNNWithoutKernel(GenFOBase):

    @property
    def bias(self):
        bias = 1.
        return -bias * torch.arange(self.n_neurons, dtype=torch.float32)

    @property
    def kernel_weight(self):
        return torch.tensor([[[0., 0.],
                              [0., 0.]],
                             [[0., 0.],
                              [0., 0.]],
                             [[0., 0.],
                              [0., 0.]],
                             [[0., 0.],
                              [0., 0.]],
                             [[0., 0.],
                              [0., 0.]]])

    @property
    def model_class(self):
        return FullyObsSigmoidSNN


class GenFullyObsSigmoidSNNWithSmallKernel(GenFullyObsSigmoidSNNWithoutKernel):

    @property
    def kernel_weight(self):
        return torch.tensor([[[-0.00000001, 0.],
                              [0., 0.]],
                             [[0., 0.],
                              [0., 0.]],
                             [[0., 0.],
                              [0., 0.]],
                             [[0., 0.],
                              [0., 0.]],
                             [[0., 0.],
                              [0., 0.]]])

class GenFullyObsSigmoidSNN(GenFullyObsSigmoidSNNWithoutKernel):

    @property
    def kernel_weight(self):
        return torch.tensor([[[-1., 1.],
                              [0., -1.]],
                             [[0., 0.],
                              [0., 0.]],
                             [[0., 0.],
                              [0., 0.]],
                             [[0., 0.],
                              [0., 0.]],
                             [[0., 0.],
                              [0., 0.]]])

# --------------

class GenDiffSigmoidSNNWithoutKernel(GenDiffPOBase):

    @property
    def kernel_weight(self):
        return torch.tensor([[[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 0., 0.]],
                             [[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 0., 0.]]])

    @property
    def bias(self):
        bias = 1.
        return -bias \
            * torch.arange(0, self.n_neurons, dtype=torch.float32)

    @property
    def model_class(self):
        return SigmoidDiffSNN


class GenDiffSigmoidSNN(GenDiffSigmoidSNNWithoutKernel):
    @property
    def kernel_weight(self):
        return torch.tensor([[[-1., 0., 0.],
                              [0., -1., 0.],
                              [0., 0., -1.]],
                             [[-0.1, 0., 1.],
                              [0., -0.1, 0.],
                              [0., 1., -0.1]]])


class GenDiffHardtanhSNN(GenDiffPOBase):

    @property
    def kernel_weight(self):
        return torch.tensor([[[-1., 0., 0.],
                              [0., -1., 0.],
                              [0., 0., -1.]],
                             [[-0.1, 0., 1.],
                              [0., -0.1, 0.],
                              [0., 1., -0.1]]])

    @property
    def bias(self):
        return torch.tensor([0.0, 1.0, 1.0])

    @property
    def model_class(self):
        return HardtanhDiffSNN


class GenHardtanhPOSNN(GenPOBase):

    @property
    def kernel_weight(self):
        return torch.tensor([[[-1., 0., 0.],
                              [0., -1., 0.],
                              [0., 0., -1.]],
                             [[-0.1, 0., 1.],
                              [0., -0.1, 0.],
                              [0., 1., -0.1]]])

    @property
    def bias(self):
        return torch.tensor([0.0, 1.0, 0.0])

    @property
    def model_class(self):
        return HardtanhPOSNN




# ---------------- trainable models ----------------

class TrainPoisson:

    @property
    def trainable_model(self):
        if not hasattr(self, '_trainable_model'):
            self._trainable_model \
                = MultivariatePoissonProcess(intensity_list=[1.]*self.n_neurons)
        return self._trainable_model

    @property
    def var_model(self):
        return None

class TrainObsPoisson:

    @property
    def trainable_model(self):
        if not hasattr(self, '_trainable_model'):
            self._trainable_model \
                = MultivariatePoissonProcess(intensity_list=[1.]*self.n_obs_neurons)
        return self._trainable_model


class TrainSNN:

    @property
    def trainable_model(self):
        if not hasattr(self, '_trainable_model'):
            connection_tensor = torch.tensor([[True] * self.n_neurons] * self.n_neurons)
            self._trainable_model = self.model_class(
                n_neurons=self.n_neurons,
                connection_tensor=connection_tensor)
        return self._trainable_model


class TrainPOSNN:

    @property
    def trainable_model(self):
        if not hasattr(self, '_trainable_model'):
            self._trainable_model = self.model_class(
                n_obs_neurons=self.n_obs_neurons,
                n_hidden_neurons=self.n_hidden_neurons,
                connection_tensor=self.connection_tensor,
                n_inducing_points=self.kernel_weight.shape[0],
                activation_kwargs={'upperbound': upperbound})
            self._trainable_model.randomize_params(1.0, 0.0)
            self._trainable_model.randomize_diagonal(-0.1, -1.0)
        return self._trainable_model

    @property
    def var_model(self):
        if not hasattr(self, '_var_model'):
            self._var_model = self.model_class(
                n_obs_neurons=self.n_obs_neurons,
                n_hidden_neurons=self.n_hidden_neurons,
                connection_tensor=self.connection_tensor,
                n_inducing_points=self.kernel_weight.shape[0],
                activation_kwargs={'upperbound': upperbound})
            self._var_model.randomize_params(1.0, 0.0)
            self._trainable_model.randomize_diagonal(-0.1, -1.0)
        return self._var_model

class TrainDiffPOSNN:

    @property
    def trainable_model(self):
        if not hasattr(self, '_trainable_model'):
            self._trainable_model = self.model_class(
                n_obs_neurons=self.n_obs_neurons,
                n_hidden_neurons=self.n_hidden_neurons,
                connection_tensor=self.connection_tensor,
                n_inducing_points=self.kernel_weight.shape[0],
                activation_kwargs={'upperbound': upperbound},
                temperature=self.temperature)
            self._trainable_model.randomize_params(1.0, 0.0)
            self._trainable_model.randomize_diagonal(-0.1, -1.0)
        return self._trainable_model

    @property
    def var_model(self):
        if not hasattr(self, '_var_model'):
            self._var_model = self.model_class(
                n_obs_neurons=self.n_obs_neurons,
                n_hidden_neurons=self.n_hidden_neurons,
                connection_tensor=self.connection_tensor,
                n_inducing_points=self.kernel_weight.shape[0],
                activation_kwargs={'upperbound': upperbound},
                temperature=self.temperature)
            self._var_model.randomize_params(1.0, 0.0)
            self._trainable_model.randomize_diagonal(-0.1, -1.0)
        return self._var_model
