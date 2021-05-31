#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "(c) Copyright IBM Corp. 2020"


from copy import deepcopy

def replace(input_dict, pop_key, new_key, new_value):
    output_dict = deepcopy(input_dict)
    output_dict.pop(pop_key)
    output_dict[new_key] = new_value
    return output_dict

diffsnn_model_kwargs = {'n_obs_neurons': 2,
                        'n_hidden_neurons': 4,
                        'connection_tensor': [[True, True, True, True, True, True],
                                              [True, True, True, True, True, True],
                                              [True, True, True, True, True, True],
                                              [True, True, True, True, True, True],
                                              [True, True, True, True, True, True],
                                              [True, True, True, True, True, True]],
                        'n_inducing_points': 2,
                        'temperature': 0.3,
                        'activation_kwargs': {'upperbound': 5.0}
}

diffsnn_model_kwargs_var_upper = replace(
    diffsnn_model_kwargs,
    'activation_kwargs',
    'activation_kwargs',
    {'@upperbound': list(float(each_upperbound) for each_upperbound in range(1, 20))})

posnn_model_data_kwargs = {'n_obs_neurons': 2,
                           'n_hidden_neurons': 4,
                           'connection_tensor': [[True, True, True, True, True, True],
                                                 [True, True, True, True, True, True],
                                                 [True, True, True, True, True, True],
                                                 [True, True, True, True, True, True],
                                                 [True, True, True, True, True, True],
                                                 [True, True, True, True, True, True]],
                           'n_inducing_points': 2,
                           'activation_kwargs': {'upperbound': 5.}}

posnn_model_kwargs = {'n_obs_neurons': 2,
                      'n_hidden_neurons': 4,
                      'connection_tensor': [[True, True, True, True, True, True],
                                            [True, True, True, True, True, True],
                                            [True, True, True, True, True, True],
                                            [True, True, True, True, True, True],
                                            [True, True, True, True, True, True],
                                            [True, True, True, True, True, True]],
                      'n_inducing_points': 2,
                      'activation_kwargs': {'upperbound': 5.0}
}

posnn_model_kwargs_var_upper = replace(
    posnn_model_kwargs,
    'activation_kwargs',
    'activation_kwargs',
    {'@upperbound': list(float(each_upperbound) for each_upperbound in range(1, 20))})


DataGeneration_params = {
    'model_name': 'SigmoidPOSNN',
    'init': ['random', 'non-random'][0],
    'seed': 0,
    'model_kwargs': posnn_model_data_kwargs,
    'model_params': {'bias': [0.0, 1.0, 0.0, 0.0],
                     'kernel_weight': [[[-1., 0., 0., 0.],
                                        [0., -1., 0., 0.],
                                        [0., 0., -1., 0.],
                                        [0., 0., 0., -1.]],
                                       [[-0.1, 1., 1, 1.],
                                        [0., -0.1, 1., 1.],
                                        [1., 1., -0.1, 1.],
                                        [0., 0., 1., -0.1]]]},
    'randomize_kernel_weight': {'low': -5.0, 'high': 5.0},
    'randomize_bias': {'low': -1.0, 'high': 1.0},
    'randomize_diagonal': {'low': -5.0, 'high': -0.1},
    'train_sample_size': 100,
    'test_sample_size': 10,
    'length': 50
}

DataGeneration_params_var_train = replace(
    DataGeneration_params,
    'train_sample_size',
    '@train_sample_size',
    [10, 20, 30, 40, 50, 75, 100, 200])


Train_params = {
    '@model_name': ['SigmoidDiffSNN', 'SigmoidPOSNN'],
    'model_kwargs': [diffsnn_model_kwargs, posnn_model_kwargs][0],
    'use_variational': False,
    'n_epochs': 10,
    'lr': 5e-2,
    'randomize_kernel_weight': {'low': -5.0, 'high': 5.0},
    'randomize_bias': {'low': -1.0, 'high': 1.0},
    'randomize_diagonal': {'low': -5.0, 'high': -0.1},
    'obj_func_kwargs': {'n_pos': 10000,
                        'n_neg': 100,
                        'n_sampling': 1,
                        'beta': 1.},
    'fit_kwargs': {'temperature_rate': 0.95}
}

Train_params_var_upper = replace(Train_params,
                                 'model_kwargs',
                                 'model_kwargs',
                                 diffsnn_model_kwargs_var_upper)


PerformanceEvaluation_params = {
    'model_name': 'SigmoidPOSNN',
    'model_kwargs': posnn_model_kwargs,
    'obj_func_kwargs': {'n_pos': 10000,
                        'n_neg': 100,
                        'n_sampling': 1,
                        'beta': 1.}
}


PlotTrainTimeMultipleRun_params = {
    'DataGeneration_params': DataGeneration_params,
    'Train_params': Train_params_var_upper,
    'data_seed': 43,
    'train_seed': 44
}


PlotTrainTime_params = {
    'x': ('Train_params', 'model_kwargs', 'activation_kwargs', 'upperbound'),
    'plot_config_list': [{'extract_list': [(('Train_params', 'model_name'), 'SigmoidPOSNN')]},
                         {'extract_list': [(('Train_params', 'model_name'), 'SigmoidDiffSNN')]}],
    'fig_config': {'xlabel': {'xlabel': r'$\bar{\lambda}$'},
                   'ylabel': {'ylabel': 'Per-epoch computation time [sec]'},
                   'legend': {'labels': ['SNN', r'$\partial$SNN'],
                              'fontsize': 24}}
}

PlotTestLossMultipleRun_params = {
    'DataGeneration_params': DataGeneration_params_var_train,
    'Train_params': Train_params,
    'PerformanceEvaluation_params': PerformanceEvaluation_params,
    'seed': 43,
    'n_trials': 24
}

PlotTestLoss_params = {
    'x': ('DataGeneration_params', 'train_sample_size'),
    'plot_config_list': [{'extract_list': [(('Train_params', 'model_name'), 'SigmoidPOSNN')]},
                         {'extract_list': [(('Train_params', 'model_name'), 'SigmoidDiffSNN')]}],
    'fig_config': {'xlabel': {'xlabel': r'\# of training examples'},
                   'ylabel': {'ylabel': 'ELBO'},
                   'legend': {'labels': ['SNN', r'$\partial$SNN']}}
}

EvaluateGradientVariance_params = Train_params
