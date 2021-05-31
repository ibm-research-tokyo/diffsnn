#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = 'Copyright IBM Corp. 2020, 2021'

from abc import abstractmethod
from typing import List
from numpy.random import default_rng
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torch
from ..data import EventSeq, MultivariateEventSeq


class PointProcess(nn.Module):

    ''' Base class for a temporal point process.
    Inherit this class, and implement `conditional_intensity` and `sample`.
    `sample` is usually implemented as mixin, whereas `conditional_intensity` requires
    additional initializations and thus should be implemented by inheriting this class.
    '''

    def __init__(self, seed=43):
        super().__init__()
        self.seed = seed

    @abstractmethod
    def conditional_intensity(self, time: float, history: EventSeq) -> float:
        pass

    @abstractmethod
    def event_impact(self) -> torch.Tensor:
        ''' This function is used in the thinning algorithm,
        which requires us to set the upperbound of the conditional intensity.
        How much the conditional intensity changes by an event.
        The conditional intensity computed exactly at the event time does not take account in the event.
        This function computes how much the conditional intensity changes right after the event.
        '''
        pass

    def __str__(self):
        output_str = ''
        for each_name, each_param in list(self.named_parameters()):
            if each_param.requires_grad:
                output_str += ' - {}:\t{}\n'.format(each_name, each_param.data)
        return output_str[1:]

    @abstractmethod
    def sample(self, history: EventSeq) -> EventSeq:
        pass

    @property
    def rng(self):
        if not hasattr(self, '_rng'):
            self._rng = default_rng(seed=self.seed)
        return self._rng

    def log_conditional_intensity(self, time: float, history: EventSeq) -> float:
        return torch.log(self.conditional_intensity(time, history))

    def draw_conditional_intensity(self,
                                   history: EventSeq,
                                   file_name: str,
                                   tmin=-float('inf'),
                                   tmax=float('inf')):
        tmin = max(tmin, history.obs_period[0])
        tmax = min(tmax, history.obs_period[1])
        x_list = np.arange(tmin, tmax, (tmax - tmin)/1000.0)
        y_list = [self.conditional_intensity(each_time, history) for each_time in x_list]
        plt.plot(x_list, y_list)
        for each_time in history.time_list:
            plt.plot(each_time, 0, marker='.')
        plt.savefig(file_name)
        plt.clf()

    def neg_ll(self, history: EventSeq, n_pos=10, n_neg=100, **kwargs) -> torch.Tensor:
        # prob of no events
        time_list = self.rng.uniform(low=history.obs_period[0],
                                     high=history.obs_period[1],
                                     size=n_neg)
        neg_ll = history.obs_len * torch.mean(torch.stack([self.conditional_intensity(
            time_list[each_idx],
            history) for each_idx in range(n_neg)]))

        # prob of events
        if len(history) != 0:
            neg_ll = neg_ll - len(history) * torch.mean(
                torch.stack([self.log_conditional_intensity(history.sample()[0], history)\
                             for _ in range(n_pos)]))
        return neg_ll + self.regularize()

    def obj_func(self, history: EventSeq, **kwargs) -> torch.Tensor:
        return self.neg_ll(history, **kwargs)

    def clamp_params(self):
        pass

    def randomize_params(self, upper, lower, except_for=[]):
        for each_param in self.params:
            if each_param not in except_for and self.params[each_param].requires_grad:
                nn.init.uniform_(self.params[each_param], lower, upper)

    def fit(self,
            history_list: List[EventSeq],
            n_epochs=100,
            obj_func_kwargs={},
            optimizer='Adagrad',
            optimizer_kwargs={'lr': 1e-2},
            shuffle=False,
            print_freq=10,
            logger=print,
            **fit_kwargs):
        optimizer = getattr(torch.optim, optimizer)(
            params=self.parameters(),
            **optimizer_kwargs)

        for iter_idx in range(n_epochs):
            if shuffle:
                self.rng.shuffle(history_list)
            running_loss = 0
            for each_history in history_list:
                loss = self.obj_func(each_history, **obj_func_kwargs)
                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.clamp_params()
            if iter_idx % print_freq == 0:
                logger('#(iter) = {}\t loss = {}'.format(iter_idx, running_loss/len(history_list)))
        return self

    def simulate(self, sample_size: int, obs_period: List[float]):
        with torch.no_grad():
            history_list = []
            for _ in range(sample_size):
                history = EventSeq([], [obs_period[0], obs_period[0]])
                history = self.sample_seq(
                    history,
                    base_intensity=self.upperbound_cond_int(history),
                    end_time=obs_period[1])
                history_list.append(history)
        return history_list

    def upperbound_cond_int(self, history:EventSeq) -> torch.tensor:
        if history.time_list:
            return (self.conditional_intensity(history.time_list[-1], history).sum()\
                    + self.event_impact())
        else:
            return (self.conditional_intensity(history.obs_period[0], history).sum()\
                    + self.event_impact())

    def regularize(self):
        return 0.


class MultivariatePointProcess(PointProcess):

    @property
    def dim(self):
        raise NotImplementedError

    def conditional_intensity(self,
                              time: float,
                              mark,
                              history: MultivariateEventSeq) -> torch.FloatTensor:
        ''' conditional intensity
        '''
        return self.all_conditional_intensity(
            time,
            history,
            dim_list=[mark.nonzero(as_tuple=False).item()])

    def log_conditional_intensity(self,
                                  time: float,
                                  mark,
                                  history: MultivariateEventSeq) -> torch.FloatTensor:
        ''' log conditional intensity
        '''
        return self.all_log_conditional_intensity(
            time,
            history,
            dim_list=[mark.nonzero(as_tuple=False).item()])

    @abstractmethod
    def all_conditional_intensity(self,
                                  time: float,
                                  history: MultivariateEventSeq,
                                  dim_list=None) -> torch.FloatTensor:
        ''' conditional intensity for all possible marks
        '''
        pass

    def all_log_conditional_intensity(self,
                                      time: float,
                                      history: MultivariateEventSeq,
                                      dim_list=None) -> torch.FloatTensor:
        ''' log conditional intensity for all possible marks
        '''
        return torch.log(self.all_conditional_intensity(time, history, dim_list=dim_list))

    def draw_conditional_intensity(self,
                                   history: MultivariateEventSeq,
                                   file_name: str,
                                   tmin=-float('inf'),
                                   tmax=float('inf')):
        tmin = max(tmin, history.obs_period[0])
        tmax = min(tmax, history.obs_period[1])
        x_list = np.arange(tmin, tmax, (tmax - tmin)/1000.0)
        y_list = torch.stack([
            self.all_conditional_intensity(each_time, history) \
            for each_time in x_list]).detach().numpy()
        fig, ax_list = plt.subplots(self.dim, 1, figsize=(10, 3*self.dim))
        for each_idx in range(self.dim):
            ax_list[each_idx].plot(x_list, y_list[:, each_idx])
            ax_list[each_idx].set_title('Conditional intensity of neuron {}'.format(each_idx))
            for each_time, each_mark in history.time_mark_list:
                if each_mark[each_idx] == 1:
                    ax_list[each_idx].plot(each_time, 0, marker='.', color='grey')
        plt.tight_layout()
        plt.savefig(file_name)
        plt.clf()

    def simulate(self, sample_size: int, obs_period: List[float]):
        history_list = []
        with torch.no_grad():
            for _ in range(sample_size):
                history = MultivariateEventSeq([], [], [obs_period[0], obs_period[0]])
                history = self.sample_seq(
                    history,
                    base_intensity=self.upperbound_cond_int(history),
                    end_time=obs_period[1])
                history_list.append(history)
        return history_list

    def upperbound_cond_int(self, history:EventSeq) -> torch.tensor:
        if history.time_list:
            return (self.all_conditional_intensity(history.time_list[-1], history).sum()\
                    + self.event_impact())
        else:
            return (self.all_conditional_intensity(history.obs_period[0], history).sum()\
                    + self.event_impact())

    def neg_ll(self, history: MultivariateEventSeq, n_pos=10, n_neg=100, **kwargs) -> torch.Tensor:
        neg_ll_pos = 0
        if len(history) != 1:
            if n_pos > len(history):
                pos_sample = history.time_mark_list
                n_pos = len(history)
            else:
                pos_sample = history.sample(n_pos)
            for each_example in pos_sample:
                neg_ll_pos = neg_ll_pos - self.log_conditional_intensity(
                    each_example[0],
                    each_example[1],
                    history)
            neg_ll_pos = (neg_ll_pos / n_pos) * len(history)
        time_list = self.rng.uniform(low=history.obs_period[0],
                                     high=history.obs_period[1],
                                     size=n_neg)
        neg_ll_neg = (history.obs_len / n_neg) \
            * torch.stack([self.all_conditional_intensity(
                time_list[each_idx],
                history
            ) for each_idx in range(n_neg)]).sum()
        return neg_ll_pos + neg_ll_neg + self.regularize()
