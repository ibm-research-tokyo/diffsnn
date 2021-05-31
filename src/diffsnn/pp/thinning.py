#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = 'Copyright IBM Corp. 2020, 2021'

from abc import abstractmethod
from copy import deepcopy
import numpy as np
import torch
from .poisson import PoissonProcess
from ..data import EventSeq, MultivariateEventSeq
from ..utils import complete_logprob


class ThinningAlgorithmMixinBase:

    ''' Base class for thinning algorithms
    '''
    def categorical(self, logits):
        uniform_rv = self.rng.uniform(size=len(logits))
        return np.argmax(logits - np.log(-np.log(uniform_rv)))

    def sample(self, history: EventSeq, base_intensity: float):
        return self.sample_seq(history,
                               base_intensity,
                               float('inf'),
                               max_event=1)

    @abstractmethod
    def sample_seq(self,
                   history: EventSeq,
                   base_intensity: float,
                   end_time: float,
                   max_event=float('inf')) -> EventSeq:
        raise NotImplementedError


class UnivariateThinningAlgorithmMixin(ThinningAlgorithmMixinBase):

    def sample_seq(self,
                   history: EventSeq,
                   base_intensity: float,
                   end_time: float,
                   max_event=float('inf')) -> EventSeq:
        n_sampled_event = 0
        if not hasattr(self, 'base_pp'):
            self.base_pp = PoissonProcess(
                intensity=base_intensity,
                seed=self.seed)
            self.base_pp.params['intensity'].requires_grad = False
        base_pp_history = deepcopy(history)
        while True:
            self.base_pp.params['intensity'].data = base_intensity
            log_base_intensity = np.log(base_intensity)
            candidate_time_stamp = self.base_pp.sample_candidate(
                base_pp_history)
            if candidate_time_stamp > end_time:
                break
            log_cond_int_tensor = self.log_conditional_intensity(
                candidate_time_stamp,
                history).reshape(-1)
            log_acceptance_rate_tensor = complete_logprob(
                log_cond_int_tensor - log_base_intensity)
            random_idx = self.categorical(log_acceptance_rate_tensor)
            if random_idx == 0:
                history.append(candidate_time_stamp)
                n_sampled_event += 1
                base_intensity = self.upperbound_cond_int(history)
                if n_sampled_event >= max_event:
                    break
            base_pp_history.append(candidate_time_stamp)
        if max_event == float('inf'):
            history.obs_period[1] = end_time
        return history


class MultivariateThinningAlgorithmMixin(ThinningAlgorithmMixinBase):

    def sample_seq(self,
                   history: MultivariateEventSeq,
                   base_intensity: float,
                   end_time: float,
                   max_event=float('inf')) -> MultivariateEventSeq:
        n_sampled_event = 0
        if not hasattr(self, 'base_pp'):
            self.base_pp = PoissonProcess(
                intensity=base_intensity,
                seed=self.seed)
            self.base_pp.params['intensity'].requires_grad = False
        base_pp_history = deepcopy(history)
        while True:
            self.base_pp.params['intensity'].data = base_intensity
            log_base_intensity = np.log(base_intensity)
            candidate_time_stamp = self.base_pp.sample_candidate(
                base_pp_history)
            if candidate_time_stamp > end_time:
                break
            with torch.no_grad():
                log_cond_int_tensor = self.all_log_conditional_intensity(
                    candidate_time_stamp,
                    history)
            log_acceptance_rate_tensor = complete_logprob(
                log_cond_int_tensor - log_base_intensity)
            random_idx = self.categorical(log_acceptance_rate_tensor)
            if random_idx != self.dim:
                mark = torch.zeros(self.dim, dtype=torch.float32)
                mark[random_idx] += 1.
                history.append(candidate_time_stamp, mark)
                n_sampled_event += 1
                base_intensity = self.upperbound_cond_int(history)
                if n_sampled_event >= max_event:
                    break
            base_pp_history.append(candidate_time_stamp, torch.tensor([]))
        if max_event == float('inf'):
            history.obs_period[1] = end_time
        return history
