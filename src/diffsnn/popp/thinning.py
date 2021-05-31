#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = 'Copyright IBM Corp. 2020, 2021'

import math
import torch
from ..data import (EventSeq,
                    MultivariateEventSeq,
                    append_hidden)
from ..utils import complete_logprob
from ..pp.poisson import PoissonProcess
from ..pp.thinning import MultivariateThinningAlgorithmMixin

EPS = 1e-2


class MultivariateThinningAlgorithmForPOMixin(MultivariateThinningAlgorithmMixin):

    def sample_hidden_seq(self,
                          history: MultivariateEventSeq,
                          base_intensity: float) -> MultivariateEventSeq:
        ''' impute hidden units
        '''
        tgt_dim_list = list(range(self.obs_dim, self.obs_dim + self.hidden_dim))
        output_history = append_hidden(history, self.hidden_dim)
        base_intensity = base_intensity + EPS
        log_base_intensity = math.log(base_intensity)
        if not hasattr(self, 'base_pp'):
            self.base_pp = PoissonProcess(
                intensity=base_intensity,
                seed=self.seed)
            self.base_pp.params['intensity'].requires_grad = False
        base_pp_history = EventSeq(time_list=[],
                                   obs_period=[history.obs_period[0],
                                               history.obs_period[0]])
        self.base_pp.params['intensity'].data = base_intensity
        while True:
            candidate_time_stamp = self.base_pp.sample_candidate(
                base_pp_history)
            if candidate_time_stamp > history.obs_period[1]:
                break
            log_cond_int_tensor = self.all_log_conditional_intensity(
                candidate_time_stamp,
                output_history,
                dim_list=tgt_dim_list)
            log_acceptance_rate_tensor = complete_logprob(
                log_cond_int_tensor - log_base_intensity)
            random_idx = self.categorical(log_acceptance_rate_tensor)
            if random_idx != self.hidden_dim:
                _mark = [0.] * self.dim
                _mark[tgt_dim_list[random_idx]] = 1.0
                mark = torch.tensor(_mark)
                output_history.insert_hidden(candidate_time_stamp,
                                             mark)
                #base_intensity = self.upperbound_cond_int(history) + EPS
            base_pp_history.append(candidate_time_stamp)
        history.obs_period[1] = history.obs_period[1]
        return output_history
