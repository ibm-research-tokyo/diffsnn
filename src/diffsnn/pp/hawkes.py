#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = 'Copyright IBM Corp. 2020, 2021'

from scipy.stats import norm
import torch
from torch import nn
from .base import PointProcess
from .thinning import UnivariateThinningAlgorithmMixin
from ..data import EventSeq


class HawkesProcess(UnivariateThinningAlgorithmMixin, PointProcess):

    def __init__(self, bg_intensity=1.0, weight=0.1, decay=1.0, seed=43):
        super().__init__(seed=seed)
        self.init_params(bg_intensity, weight, decay)

    def init_params(self, bg_intensity: float, weight: float, decay: float):
        if not (bg_intensity > 0 and weight > 0 and decay > 0):
            raise ValueError('parameters must be positive.')
        self.params = nn.ParameterDict()
        self.params['bg_intensity'] = nn.Parameter(torch.tensor(bg_intensity),
                                                   requires_grad=True)
        self.params['weight'] = nn.Parameter(torch.tensor(weight),
                                             requires_grad=True)
        self.params['decay'] = nn.Parameter(torch.tensor(decay),
                                            requires_grad=True)

    def conditional_intensity(self, time: float, history: EventSeq) -> torch.Tensor:
        conditional_intensity = self.params['bg_intensity']
        batch_time = history.batch_time(time)
        if batch_time:
            conditional_intensity \
                = conditional_intensity \
                + self.params['weight']\
                * (torch.exp(- self.params['decay'] * (time - torch.tensor(batch_time))).sum())
        return conditional_intensity

    def event_impact(self):
        return self.params['weight']

    def neg_ll(self, history: EventSeq, **kwargs) -> torch.Tensor:
        log_numerator = 0
        log_trace = -torch.tensor(float('inf'))
        for each_idx, each_time in enumerate(history.time_list):
            log_numerator \
                = log_numerator \
                + torch.logaddexp(torch.log(self.params['bg_intensity']),
                                  torch.log(self.params['weight'])\
                                  + log_trace)
            log_trace = (-self.params['decay'] * history.td_list[each_idx+1]) \
                + torch.logaddexp(torch.tensor(0.), log_trace)
        log_denominator = self.params['bg_intensity'] * history.obs_len\
            + (self.params['weight'] / self.params['decay']) * (len(history) - torch.exp(log_trace))
        return - log_numerator + log_denominator

    def neg_ll_wo_recursion(self, history: EventSeq, **kwargs) -> torch.Tensor:
        log_numerator = 0
        for each_time in history.time_list:
            log_numerator\
                = log_numerator + torch.log(self.conditional_intensity(each_time, history))
        potential = torch.exp(-self.params['decay'] \
                              * (history.obs_period[1] - torch.tensor(history.time_list))).sum()\
                              - len(history)
        log_denominator = self.params['bg_intensity'] * history.obs_len \
            - (self.params['weight'] / self.params['decay']) * potential
        return - log_numerator + log_denominator

    def clamp_params(self):
        for each_param in self.parameters():
            each_param.data.clamp_(1e-8)


class NonparametricHawkesProcessWithStochasticApproximation(UnivariateThinningAlgorithmMixin, PointProcess):

    def __init__(self,
                 bg_intensity=1.0,
                 n_inducing_points=5,
                 scale=1.0,
                 low=0.1,
                 high=10.0,
                 seed=43):
        super().__init__(seed=seed)
        self.scale = scale
        self.init_params(bg_intensity, n_inducing_points, low, high)

    def init_params(self, bg_intensity: float, n_inducing_points: int, low: float, high: float):
        if not (bg_intensity > 0 and isinstance(n_inducing_points, int) and 0. <= low < high):
            raise ValueError('invalid parameters')
        self.params = nn.ParameterDict()
        self.params['bg_intensity'] = nn.Parameter(torch.tensor(bg_intensity),
                                                   requires_grad=True)
        self.params['kernel_weight'] = nn.Parameter(torch.zeros(n_inducing_points),
                                                    requires_grad=True)
        self.params['ind_points'] = nn.Parameter(
            torch.linspace(low,
                           high,
                           n_inducing_points), requires_grad=False)

    def filter_func(self, time, event_time) -> torch.FloatTensor:
        if time >= event_time:
            pdf_tensor = torch.tensor(norm.pdf(self.params['ind_points'],
                                               loc=time-event_time,
                                               scale=self.scale)).to(torch.float32)
            #pdf_tensor = (pdf_tensor > 1e-8) * pdf_tensor
            filter_val = self.params['kernel_weight'] @ pdf_tensor
        else:
            filter_val = torch.tensor(0.)
        return filter_val

    def filter_history(self, time, history: EventSeq) -> torch.FloatTensor:
        batch_time_list = history.batch_time(time)
        if batch_time_list:
            pdf_tensor = torch.tensor(norm.pdf(
                torch.tensor(batch_time_list).reshape(-1, 1) \
                + self.params['ind_points'].reshape(-1),
                loc=time,
                scale=self.scale)).to(torch.float32)
            #pdf_tensor = (pdf_tensor > 1e-8) * pdf_tensor # initially necessary to avoid nan
            #filter_val = pdf_tensor @ self.params['kernel_weight'] # this yields nan
            if torch.isnan(self.params['kernel_weight']).any():
                raise RuntimeError('parameter becomes nan')
            filter_val = pdf_tensor * self.params['kernel_weight'] # this does not yield nan
            filter_val = filter_val.sum()
        else:
            filter_val = torch.tensor(0.)
        return filter_val

    def conditional_intensity(self, time: float, history: EventSeq) -> torch.Tensor:
        return self.params['bg_intensity'] + self.filter_history(time, history)

    def _slow_conditional_intensity(self, time: float, history: EventSeq) -> torch.Tensor:
        ''' !!! only for testing purposes !!!
        '''
        conditional_intensity = self.params['bg_intensity']
        batch_time_list = history.batch_time(time)
        for each_event_time in batch_time_list:
            conditional_intensity \
                = conditional_intensity \
                + self.filter_func(time, each_event_time)
        return conditional_intensity

    def event_impact(self) -> torch.FloatTensor:
        return self.params['kernel_weight'].sum() * norm.pdf(0)

    def clamp_params(self):
        for each_param in self.parameters():
            each_param.data.clamp_(0.)


class NonparametricHawkesProcess(NonparametricHawkesProcessWithStochasticApproximation):

    def neg_ll(self, history: EventSeq, **kwargs) -> torch.Tensor:
        # compute numerator
        log_numerator = 0
        for each_idx, each_time in enumerate(history.time_list):
            log_numerator \
                = log_numerator \
                + torch.log(self.conditional_intensity(each_time, history))

        # compute denominator
        terminal_cdf_tensor = torch.tensor(
            norm.cdf(history.obs_period[1] \
                     - (torch.tensor(history.time_list).reshape(-1,1)\
                        + self.params['ind_points'].reshape(-1)),
                     loc=0,
                     scale=self.scale)).to(torch.float32)
        start_cdf_tensor = torch.tensor(
            norm.cdf(-self.params['ind_points'].reshape(-1),
                     scale=self.scale)).to(torch.float32)
        diff_cdf_tensor = (terminal_cdf_tensor - start_cdf_tensor)
        #diff_cdf_tensor = (diff_cdf_tensor > 1e-8) * diff_cdf_tensor
        log_denominator = self.params['bg_intensity'] * history.obs_len \
            + (diff_cdf_tensor * self.params['kernel_weight']).sum()

        # for debugging
        if 'debug' in kwargs:
            if kwargs['debug']:
                _log_denominator = self.params['bg_intensity'] * history.obs_len
                for each_idx, each_inducing_point in enumerate(self.params['ind_points'].reshape(-1)):
                    for each_event_time in history.time_list:
                        _log_denominator = _log_denominator + self.params['kernel_weight'][each_idx] \
                            * (norm.cdf(history.obs_period[1], loc=each_event_time+each_inducing_point.item(),scale=self.scale)\
                               - norm.cdf(each_event_time, loc=each_event_time+each_inducing_point.item(),scale=self.scale))
                if not (torch.allclose(log_denominator, _log_denominator, atol=1e-4)):
                    raise RuntimeError('log_denominator computation has bugs')
        return - log_numerator + log_denominator
