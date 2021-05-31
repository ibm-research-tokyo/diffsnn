#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = 'Copyright IBM Corp. 2020, 2021'

from typing import List
import torch
from torch.nn import Parameter, ParameterDict
from .base import PointProcess, MultivariatePointProcess
from ..data import EventSeq, MultivariateEventSeq
from ..utils import idx2onehot

class PoissonProcess(PointProcess):

    def __init__(self, intensity: float, seed=43):
        super().__init__(seed)
        self.init_params(intensity)

    def init_params(self, intensity: float):
        if not intensity > 0:
            raise ValueError('parameters must be positive.')
        self.params = ParameterDict()
        self.params['intensity'] = Parameter(torch.tensor(float(intensity)),
                                             requires_grad=True)

    def conditional_intensity(self, time: float, history: EventSeq) -> torch.Tensor:
        return self.params['intensity']

    def event_impact(self):
        return 0

    def sample(self, history: EventSeq, **kwargs) -> EventSeq:
        return history.append(self.sample_candidate(history))

    def sample_seq(self,
                   history: EventSeq,
                   end_time: float,
                   max_event=float('inf'),
                   **kwargs) -> EventSeq:
        n_sampled_event = 0
        while True:
            candidate_time_stamp = self.sample_candidate(history)
            if candidate_time_stamp > end_time:
                break
            history.append(candidate_time_stamp)
            n_sampled_event += 1
            if n_sampled_event >= max_event:
                break
        if max_event == float('inf'):
            history.obs_period[1] = end_time
        return history

    def sample_candidate(self, history: EventSeq) -> float:
        return history.obs_period[1] \
            + self.rng.exponential(scale=1.0/float(self.params['intensity']))

    def neg_ll(self, history: EventSeq, **kwargs) -> torch.Tensor:
        return - len(history) * torch.log(self.params['intensity']) \
            + (history.obs_period[1] - history.obs_period[0]) * self.params['intensity']


class MultivariatePoissonProcess(MultivariatePointProcess):

    def __init__(self, intensity_list: List[float], seed=43):
        super().__init__(seed)
        self.init_params(intensity_list)

    @property
    def dim(self):
        return len(self.params['intensity_list'])

    def init_params(self, intensity_list: List[float]):
        if not all(torch.tensor(intensity_list) > 0):
            raise ValueError('parameters must be positive.')
        self.params = ParameterDict()
        self.params['intensity_list'] =\
            Parameter(torch.tensor(intensity_list),
                      requires_grad=True)

    def all_conditional_intensity(self,
                                  time: float,
                                  history: MultivariateEventSeq,
                                  dim_list=None) -> torch.FloatTensor:
        if dim_list is None:
            return self.params['intensity_list']
        else:
            return self.params['intensity_list'][dim_list]

    def conditional_intensity(self,
                              time: float,
                              mark,
                              history: MultivariateEventSeq) -> torch.Tensor:
        return self.params['intensity_list'][mark.nonzero(as_tuple=False).item()]

    def event_impact(self):
        return 0

    def sample(self, history: MultivariateEventSeq, **kwargs) -> EventSeq:
        return history.append(*self.sample_candidate(history))

    def sample_candidate(self, history: MultivariateEventSeq) -> float:
        sampled_mark = torch.tensor(self.rng.multinomial(
            n=1,
            pvals=(self.params['intensity_list']/(self.params['intensity_list'].sum())).detach().numpy()))
        return (history.obs_period[1] + self.rng.exponential(scale=float(1.0/(self.params['intensity_list'].sum()))),
                sampled_mark)

    def sample_seq(self,
                   history: EventSeq,
                   end_time: float,
                   max_event=float('inf'),
                   **kwargs) -> EventSeq:
        n_sampled_event = 0
        while True:
            candidate_time_stamp, mark = self.sample_candidate(history)
            if candidate_time_stamp > end_time:
                break
            history.append(candidate_time_stamp, mark)
            n_sampled_event += 1
            if n_sampled_event >= max_event:
                break
        if max_event == float('inf'):
            history.obs_period[1] = end_time
        return history

    def neg_ll(self, history: EventSeq, **kwargs) -> torch.Tensor:
        return -history.mark_tensor.sum(axis=0).to(torch.float32) \
            @ torch.log(self.params['intensity_list'])\
            + history.obs_len * self.params['intensity_list'].sum()
