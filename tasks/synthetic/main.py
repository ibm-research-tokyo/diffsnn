#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A template main script.
"""

__author__ = "Hiroshi Kajino"
__copyright__ = "Copyright IBM Corp. 2020, 2021"

# set luigi_config_path BEFORE importing luigi
import os
from pathlib import Path
import sys
from luigine.abc import (AutoNamingTask,
                         main,
                         MultipleRunBase,
                         LinePlotMultipleRun)

from copy import deepcopy
from datetime import datetime
from time import time
import glob
import logging
from luigi.util import requires
import luigi
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diffsnn.pp.poisson import MultivariatePoissonProcess
from diffsnn.pp.snn import FullyObsSigmoidSNN, FullyObsHardtanhSNN
from diffsnn.popp.snn import HardtanhPOSNN, SigmoidPOSNN
from diffsnn.diffpopp.snn import (HardtanhDiffSNN,
                                  SigmoidDiffSNN,
                                  HeuristicHardtanhDiffSNN,
                                  HeuristicSigmoidDiffSNN)
from diffsnn.data import delete_hidden

try:
    working_dir = Path(sys.argv[1:][sys.argv[1:].index("--working-dir")
                                    + 1]).resolve()
except ValueError:
    raise ValueError("--working-dir option must be specified.")

# load parameters from `INPUT/param.py`
sys.path.append(str((working_dir / 'INPUT').resolve()))
from param import (DataGeneration_params,
                   Train_params,
                   PerformanceEvaluation_params,
                   EvaluateGradientVariance_params,
                   PlotTrainTime_params,
                   PlotTrainTimeMultipleRun_params,
                   PlotTestLoss_params,
                   PlotTestLossMultipleRun_params)

logger = logging.getLogger('luigi-interface')
AutoNamingTask._working_dir = working_dir
AutoNamingTask.working_dir = luigi.Parameter(default=str(working_dir))

# ----------- preamble ------------


model_dict = {'MultivariatePoissonProcess': MultivariatePoissonProcess,
              'FullyObsSigmoidSNN': FullyObsSigmoidSNN,
              'FullyObsHardtanhSNN': FullyObsHardtanhSNN,
              'HardtanhPOSNN': HardtanhPOSNN,
              'HardtanhDiffSNN': HardtanhDiffSNN,
              'HeuristicHardtanhDiffSNN': HeuristicHardtanhDiffSNN,
              'SigmoidPOSNN': SigmoidPOSNN,
              'SigmoidDiffSNN': SigmoidDiffSNN,
              'HeuristicSigmoidDiffSNN': HeuristicSigmoidDiffSNN}


def dict_param2dict(dict_param, prefix=''):
    output_dict = {}
    for each_key in dict_param:
        if isinstance(
                dict_param[each_key],
                (dict, luigi.freezing.FrozenOrderedDict)):
            dict_repl = dict_param2dict(
                dict_param[each_key],
                prefix=prefix + '_' + each_key if prefix != '' else each_key)
            output_dict.update(dict_repl)
        elif isinstance(dict_param[each_key], (list, tuple)):
            pass
        else:
            output_dict[prefix + '_' + each_key] = dict_param[each_key]
    return output_dict


def get_initial_model(params, train_seed):
    model = model_dict[params['model_name']](
        seed=train_seed,
        **params['model_kwargs'])
    model.randomize_params(params['randomize_kernel_weight']['high'],
                           params['randomize_kernel_weight']['low'],
                           except_for=['bias'])
    model.randomize_params(params['randomize_bias']['high'],
                           params['randomize_bias']['low'],
                           except_for=['kernel_weight'])
    model.randomize_diagonal(params['randomize_diagonal']['high'],
                             params['randomize_diagonal']['low'])
    return model


# Define tasks

class DataGeneration(AutoNamingTask):

    DataGeneration_params = luigi.DictParameter()
    data_seed = luigi.IntParameter()

    def requires(self):
        return []

    def run_task(self, input_list):
        torch.manual_seed(self.data_seed)
        if self.DataGeneration_params.get('init', 'non-random') == 'random':
            gen_model = get_initial_model(
                self.DataGeneration_params,
                self.data_seed)
        elif self.DataGeneration_params.get('init', 'non-random') == 'non-random':
            gen_model = model_dict[self.DataGeneration_params['model_name']](
                seed=self.data_seed,
                **self.DataGeneration_params['model_kwargs'])
            gen_model.params['bias'].data \
                = torch.tensor(self.DataGeneration_params['model_params']['bias'])
            gen_model.params['kernel_weight'].data \
                = torch.tensor(self.DataGeneration_params['model_params']['kernel_weight'])
        else:
            raise ValueError('initialization can be either random or non-random')

        train_history_list = gen_model.simulate(self.DataGeneration_params['train_sample_size'],
                                                [0, self.DataGeneration_params['length']])
        train_po_history_list = [delete_hidden(
            each_history,
            self.DataGeneration_params['model_kwargs']['n_obs_neurons']) \
                                 for each_history in train_history_list]

        test_history_list = gen_model.simulate(self.DataGeneration_params['test_sample_size'],
                                               [0, self.DataGeneration_params['length']])
        test_po_history_list = [delete_hidden(
            each_history,
            self.DataGeneration_params['model_kwargs']['n_obs_neurons']) \
                                for each_history in test_history_list]

        gen_model.base_pp = None
        return [train_history_list,
                train_po_history_list,
                test_history_list,
                test_po_history_list,
                gen_model.state_dict()]


@requires(DataGeneration)
class Train(AutoNamingTask):

    output_ext = luigi.Parameter(default='pth')
    Train_params = luigi.DictParameter()
    train_seed = luigi.IntParameter()

    def run_task(self, input_list):
        torch.manual_seed(self.train_seed)
        _, train_po_history_list, _, _, _ = input_list[0]

        train_model = get_initial_model(self.Train_params,
                                        self.train_seed)
        var_model = get_initial_model(self.Train_params,
                                      self.train_seed+1)

        train_model.fit(train_po_history_list,
                        variational_dist=var_model,
                        use_variational=self.Train_params['use_variational'],
                        n_epochs=self.Train_params['n_epochs'],
                        optimizer_kwargs={'lr': self.Train_params['lr']},
                        obj_func_kwargs=self.Train_params['obj_func_kwargs'],
                        logger=logger.info,
                        print_freq=max(self.Train_params['n_epochs'] // 10, 1),
                        **self.Train_params.get('fit_kwargs', {}))
        return train_model, var_model

    def save_output(self, res):
        train_model, var_model = res

        var_model.base_pp = None
        torch.save(var_model.state_dict(),
                   self.output().path.replace('.pth', '_var.pth'))

        train_model.base_pp = None
        torch.save(train_model.state_dict(),
                   self.output().path)


    def load_output(self):
        state_dict = torch.load(self.output().path)
        train_model = model_dict[self.Train_params['model_name']](
            **self.Train_params['model_kwargs'])
        train_model.load_state_dict(state_dict)

        state_dict = torch.load(self.output().path.replace('.pth', '_var.pth'))
        var_model = model_dict[self.Train_params['model_name']](
            **self.Train_params['model_kwargs'])
        var_model.load_state_dict(state_dict)
        return train_model, var_model


@requires(DataGeneration)
class CalculateTrainTime(AutoNamingTask):

    Train_params = luigi.DictParameter()
    train_seed = luigi.IntParameter()

    def run_task(self, input_list):
        torch.manual_seed(self.train_seed)
        _, train_po_history_list, _, _, _ = input_list[0]

        train_model = get_initial_model(self.Train_params,
                                        self.train_seed)
        var_model = get_initial_model(self.Train_params,
                                      self.train_seed+1)
        start_time = time()
        train_model.fit(train_po_history_list,
                        variational_dist=var_model,
                        use_variational=self.Train_params['use_variational'],
                        n_epochs=self.Train_params['n_epochs'],
                        optimizer_kwargs={'lr': self.Train_params['lr']},
                        obj_func_kwargs=self.Train_params['obj_func_kwargs'],
                        logger=logger.info,
                        print_freq=max(self.Train_params['n_epochs'] // 10, 1),
                        **self.Train_params.get('fit_kwargs', {}))
        end_time = time()
        return (end_time - start_time) / self.Train_params['n_epochs']


class CollectTrainTime(MultipleRunBase):

    MultipleRun_params = luigi.DictParameter()
    score_name = luigi.Parameter(default='Computation time')

    def obj_task(self, **kwargs):
        return CalculateTrainTime(**kwargs)


class CollectTestLoss(MultipleRunBase):

    MultipleRun_params = luigi.DictParameter()
    score_name = luigi.Parameter(default='Test loss')

    def obj_task(self, **kwargs):
        return PerformanceEvaluation(**kwargs)


class PerformanceEvaluation(AutoNamingTask):

    DataGeneration_params = luigi.DictParameter(
        default=DataGeneration_params)
    Train_params = luigi.DictParameter(
        default=Train_params)
    seed = luigi.IntParameter()
    n_trials = luigi.IntParameter()
    #use_mlflow = luigi.BoolParameter(default=False)
    PerformanceEvaluation_params = luigi.DictParameter(
        default=PerformanceEvaluation_params)

    def requires(self):
        np.random.seed(self.seed)
        data_seed_list = np.random.randint(4294967295, size=self.n_trials)
        train_seed_list = np.random.randint(4294967295, size=self.n_trials)
        eval_seed_list = np.random.randint(4294967295, size=self.n_trials)
        logger.info(' * data seed: {}'.format(data_seed_list))
        logger.info(' * train seed: {}'.format(train_seed_list))
        logger.info(' * eval seed: {}'.format(eval_seed_list))
        return [SinglePerformanceEvaluation(
            DataGeneration_params=self.DataGeneration_params,
            Train_params=self.Train_params,
            PerformanceEvaluation_params=self.PerformanceEvaluation_params,
            data_seed=data_seed_list[each_idx],
            train_seed=train_seed_list[each_idx],
            eval_seed=eval_seed_list[each_idx],
            use_mlflow=self.use_mlflow)
                for each_idx in range(self.n_trials)]

    def run_task(self, input_list):
        res_df = pd.DataFrame(input_list)
        res_df.plot.hist(bins=max(len(res_df) // 10, 10))
        plt.savefig(self.output().path.replace('.pklz', '.png'))
        plt.clf()
        #self.mlflow.log_artifact(self.output().path.replace('.pklz', '.png'))
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        logger.info(str(res_df.describe()))
        #return res_df
        return -res_df.mean()['test_loss']


@requires(DataGeneration, Train)
class SinglePerformanceEvaluation(AutoNamingTask):

    ''' Performance evaluation on the validation set.
    '''

    PerformanceEvaluation_params = luigi.DictParameter()
    eval_seed = luigi.IntParameter()
    #use_mlflow = luigi.BoolParameter(default=True)

    def run_task(self, input_list):
        torch.manual_seed(self.eval_seed)
        _, train_po_history_list, \
            _, test_po_history_list, true_model_state_dict = input_list[0]
        train_model, var_model = input_list[1]
        train_model.hard = True
        var_model.hard = True

        # ----- evaluation model construction -----
        eval_model = model_dict[
            self.PerformanceEvaluation_params['model_name']](
            seed=self.eval_seed,
            **self.PerformanceEvaluation_params['model_kwargs'])
        eval_model.transfer_model(train_model)
        '''
        for each_param in eval_model.params:
            eval_model.params[each_param].data = deepcopy(train_model.params[each_param].data)
        '''

        if self.Train_params['use_variational']:
            eval_var_model = model_dict[self.PerformanceEvaluation_params['model_name']](
                seed=self.eval_seed,
                **self.PerformanceEvaluation_params['model_kwargs'])
            eval_var_model.transfer_model(var_model)
            '''
            for each_param in eval_var_model.params:
                eval_var_model.params[each_param].data = deepcopy(var_model.params[each_param].data)
            '''
        else:
            eval_var_model = eval_model

        # ----- ground truth model construction -----
        true_model = model_dict[self.DataGeneration_params['model_name']](
            seed=self.data_seed,
            **self.DataGeneration_params['model_kwargs'])
        true_model.load_state_dict(true_model_state_dict)
        '''
        true_model.params['bias'].data \
            = true_model_state_dict['params.bias']
        true_model.params['kernel_weight'].data \
            = true_model_state_dict['params.kernel_weight']
        '''

        # ----- mlflow setting -----
        '''
        self.mlflow.set_tags({'DataGeneration_' + each_item[0]: each_item[1] \
                              for each_item in self.DataGeneration_params.items()})
        self.mlflow.set_tags({'Train_' + each_item[0]: each_item[1] \
                              for each_item in self.Train_params.items()})
        self.mlflow.set_tags({'PerformanceEvaluation_' + each_item[0]: each_item[1] \
                              for each_item in self.PerformanceEvaluation_params.items()})
        self.mlflow.log_params(dict_param2dict(
            self.DataGeneration_params,
            'DataGeneration'))
        self.mlflow.log_params(dict_param2dict(
            self.Train_params,
            'Train'))
        self.mlflow.log_params(dict_param2dict(
            self.PerformanceEvaluation_params,
            'PerformanceEvaluation'))
        '''
        # ----- computing elbo -----
        train_neg_elbo = self.compute_neg_elbo(
            eval_model,
            eval_var_model,
            train_po_history_list,
            suffix='_train')
        test_neg_elbo = self.compute_neg_elbo(
            eval_model,
            eval_var_model,
            test_po_history_list,
            suffix='_test')
        true_train_neg_elbo = self.compute_neg_elbo(
            true_model,
            true_model,
            train_po_history_list,
            suffix='_true_train')
        true_test_neg_elbo = self.compute_neg_elbo(
            true_model,
            true_model,
            test_po_history_list,
            suffix='_true_test')
        logger.info(' * train_neg_elbo: {}'.format(train_neg_elbo))
        logger.info(' * test_neg_elbo: {}'.format(test_neg_elbo))
        logger.info(' * true_train_neg_elbo: {}'.format(true_train_neg_elbo))
        logger.info(' *true_ test_neg_elbo: {}'.format(true_test_neg_elbo))
        '''
        self.mlflow.log_metric('train_loss', train_neg_elbo)
        self.mlflow.log_metric('test_loss', test_neg_elbo)
        self.mlflow.log_metric('true_train_loss', true_train_neg_elbo)
        self.mlflow.log_metric('true_test_loss', true_test_neg_elbo)
        '''
        res_dict = {
            'train_loss': train_neg_elbo,
            'test_loss': test_neg_elbo,
            'true_train_loss': true_train_neg_elbo,
            'true_test_loss': true_test_neg_elbo}
        return res_dict

    def compute_neg_elbo(self, train_model, var_model, po_history_list, suffix=''):
        neg_elbo_list = []
        with torch.no_grad():
            for each_po_history in po_history_list:
                neg_elbo_list.append(
                    train_model.neg_elbo(
                        each_po_history,
                        var_model,
                        **self.PerformanceEvaluation_params['obj_func_kwargs']))
        plt.hist(neg_elbo_list, bins=max(len(neg_elbo_list) // 10, 10))
        plt.savefig(self.output().path.replace('.pklz', suffix + '.png'))
        plt.clf()
        #self.mlflow.log_artifact(self.output().path.replace('.pklz', suffix + '.png'))
        return np.mean(neg_elbo_list)


@requires(DataGeneration)
class EvaluateGradientVariance(AutoNamingTask):

    '''
    Evaluate the variance of gradient estimators.
    '''

    EvaluateGradientVariance_params = luigi.DictParameter()
    train_seed = luigi.IntParameter()

    def run_task(self, input_list):
        _, train_po_history_list, _, _, _ = input_list[0]

        train_model = get_initial_model(self.EvaluateGradientVariance_params,
                                        self.train_seed)
        if self.EvaluateGradientVariance_params['use_variational']:
            var_model = get_initial_model(self.EvaluateGradientVariance_params,
                                          self.train_seed+1)
        else:
            var_model = train_model

        grad_list_dict = {}
        for each_param in train_model.params:
            if train_model.params[each_param].requires_grad:
                grad_list_dict[each_param] = []
        for _ in range(self.EvaluateGradientVariance_params['n_epochs']):
            obj_func = 0
            for each_history in train_po_history_list:
                obj_func = obj_func \
                    + train_model.e_step_obj_func(
                        each_history,
                        variational_dist=var_model,
                        **self.EvaluateGradientVariance_params['obj_func_kwargs'])
            obj_func.backward()
            for each_param in train_model.params:
                if train_model.params[each_param].requires_grad:
                    grad_list_dict[each_param].append(
                        deepcopy(train_model.params[each_param].grad.reshape(-1)))
            train_model.zero_grad()

        return grad_list_dict

# main tasks

@requires(EvaluateGradientVariance)
class AnalyzeGradientVariance(AutoNamingTask):

    DataGeneration_params = luigi.DictParameter(default=DataGeneration_params)
    EvaluateGradientVariance_params = luigi.DictParameter(
        default=EvaluateGradientVariance_params)
    data_seed = luigi.IntParameter(default=43)
    train_seed = luigi.IntParameter(default=44)

    def run_task(self, input_list):
        grad_list_dict = input_list[0]
        cov_dict = {}
        std_list = []
        for each_param in grad_list_dict:
            grad = torch.stack(grad_list_dict[each_param]).detach().numpy()
            cov_dict[each_param] = np.cov(grad.T)
            diag_std = cov_dict[each_param].diagonal() ** 0.5
            #logger.info(f' ** diag cov of {each_param} **')
            #logger.info(str(cov_dict[each_param].diagonal()))
            logger.info(f' ** diag std of {each_param} **')
            logger.info(str(diag_std))
            std_list = std_list + (list(diag_std[np.where(diag_std > 0)]))
        logger.info(' * E[std] = {}'.format(np.mean(std_list)))
        return cov_dict

@requires(CollectTrainTime)
class PlotTrainTime(LinePlotMultipleRun):

    MultipleRun_params = luigi.DictParameter(default=PlotTrainTimeMultipleRun_params)
    LinePlotMultipleRun_params = luigi.DictParameter(default=PlotTrainTime_params)


@requires(CollectTestLoss)
class PlotTestLoss(LinePlotMultipleRun):

    MultipleRun_params = luigi.DictParameter(default=PlotTestLossMultipleRun_params)
    LinePlotMultipleRun_params = luigi.DictParameter(default=PlotTestLoss_params)


if __name__ == "__main__":
    for each_engine_status in glob.glob(str(working_dir / 'engine_status.*')):
        os.remove(each_engine_status)
    with open(working_dir / 'engine_status.ready', 'w') as f:
        f.write("ready: {}\n".format(datetime.now().strftime('%Y/%m/%d %H:%M:%S')))
    main(working_dir)
