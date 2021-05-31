#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = 'Copyright IBM Corp. 2020, 2021'

from typing import List
import torch
from ..pp.base import MultivariatePointProcess
from ..data import MultivariateEventSeq


class POMultivariatePointProcess(MultivariatePointProcess):

    ''' Multivariate point process where some of the dimensions are unobservable.
    '''

    @property
    def obs_dim(self):
        raise NotImplementedError

    @property
    def hidden_dim(self):
        raise NotImplementedError

    def e_step(self,
               history,
               variational_dist,
               optimizer,
               **obj_func_kwargs):
        ''' update variational_dist and return the value of the obj func
        '''
        loss = self.e_step_obj_func(history,
                                    variational_dist,
                                    **obj_func_kwargs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        variational_dist.clamp_params()
        return loss

    def e_step_obj_func(self,
                        history,
                        variational_dist,
                        **kwargs):
        ''' used for REINFORCE estimator.
        the gradient of this function w.r.t. the parameters corresponds to that of ELBO.
        '''
        obj = 0
        n_sampling = kwargs.get('n_sampling', 1)
        for _ in range(n_sampling):
            with torch.no_grad():
                history_with_hidden = variational_dist.sample_hidden_seq(
                    history=history,
                    base_intensity=self.upperbound_cond_int(
                        history,
                        dim=self.hidden_dim))
            neg_ll = self.neg_ll(history_with_hidden, **kwargs)
            entropy = variational_dist.hidden_entropy(history_with_hidden, **kwargs)
            obj = obj \
                - float(neg_ll.item() - entropy.item()) * entropy \
                + (neg_ll - entropy)
        return obj

    def m_step(self,
               history,
               variational_dist,
               optimizer,
               **obj_func_kwargs):
        loss = self.m_step_obj_func(history,
                                    variational_dist=variational_dist,
                                    **obj_func_kwargs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.clamp_params()
        return loss

    def m_step_obj_func(self,
                        history: MultivariateEventSeq,
                        variational_dist=None,
                        **kwargs) -> torch.Tensor:
        ''' compute a part of negation of ELBO; E_q[-log p]
        '''
        obj = 0
        n_sampling = kwargs.get('n_sampling', 1)
        for _ in range(n_sampling):
            with torch.no_grad():
                history_with_hidden = variational_dist.sample_hidden_seq(
                    history=history,
                    base_intensity=self.upperbound_cond_int(
                        history,
                        dim=self.hidden_dim))
            obj = obj \
                + self.neg_ll(history_with_hidden, **kwargs) \
                - variational_dist.hidden_entropy(history_with_hidden, **kwargs)
        return obj / n_sampling + self.regularize()

    def neg_elbo(self,
                 history: MultivariateEventSeq,
                 variational_dist=None,
                 **kwargs) -> torch.Tensor:
        ''' compute negation of ELBO; E_q[-log p + log q]]
        '''
        obj = 0
        n_sampling = kwargs.get('n_sampling', 1)
        with torch.no_grad():
            for _ in range(n_sampling):
                history_with_hidden = variational_dist.sample_hidden_seq(
                    history=history,
                    base_intensity=self.upperbound_cond_int(
                        history,
                        dim=self.hidden_dim))
                obj = obj \
                    + self.neg_ll(history_with_hidden, **kwargs) \
                    - variational_dist.hidden_entropy(history_with_hidden, **kwargs)
        return (obj / n_sampling + self.regularize()).item()

    def neg_ll(self,
               history_with_hidden: MultivariateEventSeq,
               n_pos=100,
               n_neg=100,
               **kwargs) -> torch.Tensor:
        '''
        -log p
        '''
        neg_ll_pos = 0
        if len(history_with_hidden) != 0:
            if n_pos > len(history_with_hidden):
                pos_sample = history_with_hidden.time_mark_list
                n_pos = len(history_with_hidden)
            else:
                pos_sample = history_with_hidden.sample(n_pos)
            for each_example in pos_sample:
                neg_ll_pos = neg_ll_pos \
                    - self.log_conditional_intensity(
                        each_example[0],
                        each_example[1],
                        history_with_hidden)
            neg_ll_pos = (neg_ll_pos / n_pos) * len(history_with_hidden)
        time_list = self.rng.uniform(low=history_with_hidden.obs_period[0],
                                     high=history_with_hidden.obs_period[1],
                                     size=n_neg)
        neg_ll_neg = (history_with_hidden.obs_len / n_neg) \
            * sum(sum([self.all_conditional_intensity(
                time_list[each_idx],
                history_with_hidden) \
                       for each_idx in range(n_neg)]))
        return neg_ll_pos + neg_ll_neg

    def hidden_entropy(self,
                       history_with_hidden: MultivariateEventSeq,
                       n_pos=100,
                       n_neg=100,
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
                neg_ll_pos = neg_ll_pos \
                    - self.log_conditional_intensity(
                        each_example[0],
                        each_example[1],
                        history_with_hidden)
            neg_ll_pos \
                = (neg_ll_pos / n_pos) * len(history_with_hidden.hidden_time_list)
        time_list = self.rng.uniform(low=history_with_hidden.obs_period[0],
                                     high=history_with_hidden.obs_period[1],
                                     size=n_neg)
        neg_ll_neg = (history_with_hidden.obs_len / n_neg) \
            * sum(sum([self.all_conditional_intensity(
                time_list[each_idx],
                history_with_hidden,
                dim_list=self.hidden_neuron_list) \
                       for each_idx in range(n_neg)]))
        return neg_ll_pos + neg_ll_neg

    def fit(self,
            history_list: List[MultivariateEventSeq],
            variational_dist=None,
            n_epochs=100,
            obj_func_kwargs={},
            optimizer='Adagrad',
            optimizer_kwargs={'lr': 1e-2},
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
                self.e_step(each_history,
                            variational_dist,
                            optimizer_var,
                            **obj_func_kwargs)
                loss = self.m_step(each_history,
                                   variational_dist,
                                   optimizer_model,
                                   **obj_func_kwargs)
                if print_freq != -1:
                    '''
                    loss = self.neg_elbo(each_history,
                                         variational_dist,
                                         **obj_func_kwargs)
                    '''
                    running_loss += loss.item()
            if print_freq > 0 and iter_idx % print_freq == 0:
                logger('#(iter) = {}\t loss = {}'.format(iter_idx, running_loss/len(history_list)))
            if print_freq == -1:
                logger('#(iter) = {}'.format(iter_idx))
        return self
