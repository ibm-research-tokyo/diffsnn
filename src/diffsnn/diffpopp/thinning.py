#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = 'Copyright IBM Corp. 2020, 2021'

import math
import numpy as np
from torch.nn.functional import gumbel_softmax
import torch
from ..utils import complete_logprob
from ..data import (EventSeq,
                    MultivariateEventSeq,
                    MultivariateDiffEventSeq,
                    append_diff_hidden)
from ..pp.poisson import PoissonProcess
from ..pp.thinning import MultivariateThinningAlgorithmMixin

EPS=1e-5


class MultivariateDiffThinningAlgorithmMixin(MultivariateThinningAlgorithmMixin):

    def sample_hidden_seq(self,
                          history: MultivariateEventSeq,
                          **kwargs) -> MultivariateDiffEventSeq:
        ''' impute hidden units and compute log_q
        '''
        return self.sample_hidden_seq_entropy(history, with_entropy=False)

    def sample_hidden_seq_entropy(self,
                                  history: MultivariateEventSeq,
                                  with_entropy=True):
        ''' impute hidden units and compute -log_q
        '''
        def _concrete_param(time_stamp, history, dim_list, log_base_intensity):
            ''' compute the parameters of the concrete distribuiton
            '''
            return complete_logprob(self.all_log_conditional_intensity(
                time_stamp,
                history,
                dim_list=dim_list) - log_base_intensity)

        base_intensity = float(self.upperbound_cond_int(dim=self.hidden_dim))
        log_base_intensity = math.log(base_intensity)
        log_base_intensity = np.logaddexp(math.log(base_intensity), math.log(EPS))
        base_intensity = math.exp(log_base_intensity)
        log_q = 0
        tgt_dim_list = list(range(self.obs_dim, self.obs_dim + self.hidden_dim))
        output_history = append_diff_hidden(history, self.hidden_dim) # log mark, (obs_dim + hidden_dim + 1) dimension
        if not hasattr(self, 'base_pp'):
            self.base_pp = PoissonProcess(intensity=base_intensity, seed=self.seed)
            self.base_pp.params['intensity'].requires_grad = False
        self.base_pp.params['intensity'].data = torch.tensor(base_intensity)
        base_pp_history = EventSeq(time_list=[],
                                   obs_period=[history.obs_period[0],
                                               history.obs_period[0]])
        while True:
            candidate_time_stamp = self.base_pp.sample_candidate(base_pp_history)
            if candidate_time_stamp > history.obs_period[1]:
                break

            log_acceptance_rate_tensor = _concrete_param(
                candidate_time_stamp,
                output_history,
                tgt_dim_list,
                log_base_intensity)

            #log_mark_tensor = torch.log(torch.tensor([0.] * (self.dim + 1)))
            log_mark_tensor = torch.tensor([-float('inf')] * (self.dim + 1))
            #log_mark_tensor = torch.log(torch.zeros(self.dim + 1)) # log mark
            if self.hard:
                # sample from gumbel "max" distribution
                mark_tensor_with_no_event = gumbel_softmax(
                    log_acceptance_rate_tensor,
                    tau=self.temperature,
                    hard=True)
                hidden_log_mark_tensor = torch.log(mark_tensor_with_no_event)
            else:
                hidden_log_mark_tensor = self.expconcrete_dist.rsample(
                    temperature=self.temperature,
                    logits=log_acceptance_rate_tensor)

            log_mark_tensor[self.obs_dim:] = hidden_log_mark_tensor
            base_pp_history.append(candidate_time_stamp)
            output_history.insert_hidden(candidate_time_stamp, log_mark_tensor)
            if with_entropy:
                log_q = log_q + log_base_intensity \
                    + self.expconcrete_dist.log_pdf(
                        hidden_log_mark_tensor,
                        temperature=self.temperature,
                        logits=log_acceptance_rate_tensor)

        output_history.obs_period[1] = history.obs_period[1]
        if with_entropy:
            return (output_history,
                    -log_q + history.obs_len * self.upperbound_cond_int(dim=self.hidden_dim))
        else:
            return output_history
