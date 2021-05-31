#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = 'Copyright IBM Corp. 2020, 2021'

from typing import List
import torch
from ..popp.base import POMultivariatePointProcess
from ..data import MultivariateLogDiffEventSeq, MultivariateEventSeq
from ..utils import ExpConcreteDistribution, complete_logprob


class MultivariateDifferentiablePointProcess(POMultivariatePointProcess):

    ''' Multivariate point process where some of the dimensions are unobservable.
    '''

    @property
    def expconcrete_dist(self):
        if not hasattr(self, '_expconcrete_dist'):
            self._expconcrete_dist = ExpConcreteDistribution(self.hidden_dim+1)
        return self._expconcrete_dist

    def e_step_obj_func(self,
                        history: MultivariateEventSeq,
                        variational_dist,
                        beta=1.0,
                        **kwargs) -> torch.Tensor:
        obj = 0
        n_sampling = kwargs.get('n_sampling', 1)
        for _ in range(n_sampling):
            # history_with_hidden has log mark
            history_with_hidden, entropy \
                = variational_dist.sample_hidden_seq_entropy(
                    history=history)
            '''
            _entropy = self.hidden_entropy(history_with_hidden, n_pos=10000)
            if (_entropy != entropy).item():
                print(entropy, _entropy)
                import pdb; pdb.set_trace()
            else:
                print('yay')
            '''
            obj = obj \
                + self.neg_ll(history_with_hidden, **kwargs) \
                - beta * entropy
        return obj / n_sampling + self.regularize()

    def neg_ll(self,
               history_with_hidden: MultivariateLogDiffEventSeq,
               n_pos=100,
               n_neg=100,
               **kwargs) -> torch.Tensor:
        '''
        -log p(obs, hidden)
        '''
        neg_ll_pos = torch.tensor([0.])
        if len(history_with_hidden) != 0:
            if n_pos > len(history_with_hidden):
                pos_sample = history_with_hidden.time_mark_list
                n_pos = len(history_with_hidden)
            else:
                pos_sample = history_with_hidden.sample(n_pos)
            for each_example in pos_sample:
                neg_ll_pos = neg_ll_pos - self.log_conditional_intensity(
                    each_example[0],
                    each_example[1], # log mark
                    history_with_hidden)

            neg_ll_pos = (neg_ll_pos / n_pos) * len(history_with_hidden)
        time_list = self.rng.uniform(low=history_with_hidden.obs_period[0],
                                     high=history_with_hidden.obs_period[1],
                                     size=n_neg)
        neg_ll_neg = (history_with_hidden.obs_len / n_neg) \
            * sum([self.all_conditional_intensity(
                time_list[each_idx],
                history_with_hidden,
                dim_list=list(range(self.obs_dim))).sum() \
                   for each_idx in range(n_neg)])
        neg_ll_neg = neg_ll_neg\
            + history_with_hidden.obs_len * self.upperbound_cond_int(dim=self.hidden_dim)
        return neg_ll_pos + neg_ll_neg

    def hidden_entropy(self,
                       history_with_hidden: MultivariateEventSeq,
                       n_pos=100,
                       **kwargs) -> torch.Tensor:
        '''
        -log q
        '''
        neg_ll_pos = 0
        if len(history_with_hidden.hidden_time_list) != 0:
            if n_pos > len(history_with_hidden):
                pos_sample = history_with_hidden.hidden_time_mark_list
                n_pos = len(history_with_hidden.hidden_time_list)
            else:
                pos_sample = history_with_hidden.sample_hidden(n_pos)
            for each_example in pos_sample: # hidden sample
                logits = self.all_log_conditional_intensity(
                    each_example[0],
                    history_with_hidden,
                    dim_list=list(range(self.obs_dim, self.obs_dim+self.hidden_dim)))
                logits = logits - torch.log(self.upperbound_cond_int(history_with_hidden,
                                                                     dim=self.hidden_dim))
                logits = complete_logprob(logits)
                neg_ll_pos = neg_ll_pos \
                    - torch.log(self.upperbound_cond_int(history_with_hidden,
                                                         dim=self.hidden_dim)) \
                    - self.expconcrete_dist.log_pdf(
                        each_example[1][self.obs_dim:],
                        self.temperature,
                        logits)
            neg_ll_pos \
                = (neg_ll_pos / n_pos) * len(history_with_hidden.hidden_time_list)
        # neg_ll_neg can be computed analytically for diffpp
        neg_ll_neg = history_with_hidden.obs_len * self.upperbound_cond_int(dim=self.hidden_dim)
        return neg_ll_pos + neg_ll_neg

    def fit(self,
            history_list: List[MultivariateEventSeq],
            variational_dist=None,
            n_epochs=100,
            obj_func_kwargs={},
            optimizer='Adagrad',
            optimizer_kwargs={'lr': 1e-2},
            temperature_rate=0.99,
            shuffle=False,
            print_freq=10,
            logger=print,
            **fit_kwargs):
        if not obj_func_kwargs.get('use_variational', False):
            variational_dist = self

        # set up optimizers
        optimizer_model = getattr(torch.optim, optimizer)(
            params=self.parameters(), **optimizer_kwargs)
        if variational_dist is not self:
            optimizer_var = getattr(torch.optim, optimizer)(
                params=variational_dist.parameters(), **optimizer_kwargs)
        else:
            optimizer_var = optimizer_model

        # training
        for iter_idx in range(n_epochs):
            if shuffle:
                self.rng.shuffle(history_list)
            running_loss = 0
            for each_history in history_list:
                loss = self.e_step(each_history,
                                   variational_dist,
                                   optimizer_var,
                                   **obj_func_kwargs)
                #if variational_dist != self:
                self.m_step(each_history,
                            variational_dist,
                            optimizer_model,
                            **obj_func_kwargs)
                if print_freq != -1:
                    '''
                    loss = self.neg_elbo(each_history,
                                         variational_dist,
                                         **obj_func_kwargs)
                    #running_loss += loss
                    '''
                    running_loss += loss.item()
            if print_freq > 0 and iter_idx % print_freq == 0:
                logger('#(iter) = {}\t loss = {}'.format(iter_idx, running_loss/len(history_list)))
                logger('\t\t temperature = {}'.format(self.temperature))
            if print_freq == -1:
                logger('#(iter) = {}\ttemperature = {}'.format(iter_idx, self.temperature))
            self.temperature = self.temperature * temperature_rate
        return self


class HeuristicMultivariateDifferentiablePointProcess(POMultivariatePointProcess):

    ''' Multivariate point process where some of the dimensions are unobservable.
    '''

    @property
    def expconcrete_dist(self):
        if not hasattr(self, '_expconcrete_dist'):
            self._expconcrete_dist = ExpConcreteDistribution(self.hidden_dim+1)
        return self._expconcrete_dist

    def e_step_obj_func(self,
                        history: MultivariateEventSeq,
                        variational_dist,
                        beta=1.0,
                        **kwargs) -> torch.Tensor:
        obj = 0
        n_sampling = kwargs.get('n_sampling', 1)
        for _ in range(n_sampling):
            # history_with_hidden has log mark
            history_with_hidden \
                = variational_dist.sample_hidden_seq(
                    history=history)

            entropy = self.hidden_entropy(history_with_hidden, **kwargs)
            obj = obj \
                + self.neg_ll(history_with_hidden, **kwargs) \
                - beta * entropy
        return obj / n_sampling + self.regularize()

    def fit(self,
            history_list: List[MultivariateEventSeq],
            variational_dist=None,
            n_epochs=100,
            obj_func_kwargs={},
            optimizer='Adagrad',
            optimizer_kwargs={'lr': 1e-2},
            temperature_rate=0.99,
            shuffle=False,
            print_freq=10,
            logger=print,
            **fit_kwargs):
        if not obj_func_kwargs.get('use_variational', False):
            variational_dist = self

        # set up optimizers
        optimizer_model = getattr(torch.optim, optimizer)(
            params=self.parameters(), **optimizer_kwargs)
        if variational_dist is not self:
            optimizer_var = getattr(torch.optim, optimizer)(
                params=variational_dist.parameters(), **optimizer_kwargs)
        else:
            optimizer_var = optimizer_model

        # training
        for iter_idx in range(n_epochs):
            if shuffle:
                self.rng.shuffle(history_list)
            running_loss = 0
            for each_history in history_list:
                loss = self.e_step(each_history,
                                   variational_dist,
                                   optimizer_var,
                                   **obj_func_kwargs)
                #if variational_dist != self:
                self.m_step(each_history,
                            variational_dist,
                            optimizer_model,
                            **obj_func_kwargs)
                if print_freq != -1:
                    '''
                    loss = self.neg_elbo(each_history,
                                         variational_dist,
                                         **obj_func_kwargs)
                    #running_loss += loss
                    '''
                    running_loss += loss.item()
            if print_freq > 0 and iter_idx % print_freq == 0:
                logger('#(iter) = {}\t loss = {}'.format(iter_idx, running_loss/len(history_list)))
                logger('\t\t temperature = {}'.format(self.temperature))
            if print_freq == -1:
                logger('#(iter) = {}\ttemperature = {}'.format(iter_idx, self.temperature))
            self.temperature = self.temperature * temperature_rate
        return self
