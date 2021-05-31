#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = 'Copyright IBM Corp. 2020, 2021'

from bisect import bisect_left
from copy import deepcopy
from typing import List, Any
from numpy.random import default_rng
import numpy as np
import torch
from .checker import is_n_d_tensor

def is_onehot(x_tensor):
    return len(torch.nonzero(x_tensor, as_tuple=False)) == 1 \
        and all(x_tensor[torch.nonzero(x_tensor, as_tuple=False)] == 1)

def is_simplex(x_tensor):
    return all(x_tensor >= 0) \
        and all(x_tensor <= 1) \
        and np.allclose(x_tensor.sum().item(), 1.0)


class EventSeq:

    def __init__(self,
                 time_list: List[float],
                 obs_period: List[float],
                 seed=43):
        self._check_time_list(time_list)
        self._check_obs_period(time_list, obs_period)

        self.time_list = time_list
        self.obs_period = obs_period
        self.seed = 43

    @property
    def rng(self):
        if not hasattr(self, '_rng'):
            self._rng = default_rng(seed=self.seed)
        return self._rng

    @property
    def time_tensor(self):
        if not hasattr(self, '_time_tensor'):
            self._time_tensor = torch.tensor(self.time_list)
        elif len(self) != len(self._time_tensor):
            self._time_tensor = torch.tensor(self.time_list)
        else:
            pass
        return self._time_tensor

    @staticmethod
    def _check_time_list(time_list: List[float]):
        if len(time_list) != 0:
            if not all(time_list[each_idx] <= time_list[each_idx+1] \
                       for each_idx in range(len(time_list)-1)):
                raise ValueError(
                    'time_list must be sorted & must not include duplicated time stamps.')
        if not all(isinstance(each_time, float) for each_time in time_list):
            raise ValueError('time_list must be a list of float values.')

    @staticmethod
    def _check_obs_period(time_list: List[float], obs_period: List[float]):
        if not (obs_period[0] <= obs_period[1] and len(obs_period) == 2):
            raise ValueError('obs_period must be a list of start time and end time.')
        if len(time_list) != 0:
            if not (obs_period[0] <= time_list[0] and obs_period[1] >= time_list[-1]):
                raise ValueError('The time period `obs_period` must include time_list.')

    def append(self, time_stamp: float, obs_time_end=float('inf')):
        if time_stamp <= self.obs_period[1]:
            raise ValueError('the new time_stamp is inconsistent with obs_period.')
        if time_stamp > obs_time_end:
            raise ValueError('time_stamp and obs_time_end are not consistent.')
        if obs_time_end < self.obs_period[1]:
            raise ValueError('obs_time_end must be later than self.obs_period[1]')

        self.time_list.append(time_stamp)
        if obs_time_end == float('inf'):
            self.obs_period[1] = time_stamp
        else:
            self.obs_period[1] = obs_time_end

        return self

    def insert(self, time_stamp: float):
        insert_idx = bisect_left(self.time_list, time_stamp)
        if insert_idx == 0 and self.obs_period[0] > time_stamp:
            self.obs_period[0] = time_stamp
        elif insert_idx == len(self.time_list) and self.obs_period[1] < time_stamp:
            self.obs_period[1] = time_stamp
        else:
            pass
        self.time_list.insert(insert_idx, time_stamp)
        return self

    def __len__(self):
        return len(self.time_list)

    @property
    def td_list(self):
        extended_time_list = [self.obs_period[0]] + self.time_list + [self.obs_period[1]]
        return [extended_time_list[each_idx + 1] - extended_time_list[each_idx]
                for each_idx in range(len(extended_time_list)-1)]

    def __add__(self, other):
        if self.obs_period != other.obs_period:
            raise ValueError('obs_period must be the same.')
        time_list = self.time_list + other.time_list
        time_list.sort()
        return EventSeq(time_list=time_list,
                        obs_period=self.obs_period)

    def sample(self, size=1):
        idx_array = self.rng.choice(len(self), size=size, replace=True)
        return [self.time_list[each_idx]\
                for each_idx in idx_array]

    def batch_time(self, time, len_window=float('inf')):
        rightmost_idx = bisect_left(self.time_list, time) - 1
        leftmost_idx = bisect_left(self.time_list, time - len_window)
        if rightmost_idx == -1:
            return [], []
        else:
            return self.time_list[leftmost_idx : rightmost_idx+1]

    @property
    def obs_len(self):
        return self.obs_period[1] - self.obs_period[0]


class MarkedEventSeq(EventSeq):

    '''
    Generic data class for marked point process
    '''

    def __init__(self,
                 time_list: List[float],
                 mark_list: List[Any],
                 obs_period: List[float]):
        if len(time_list) != len(mark_list):
            raise ValueError('time_list and mark_list are inconsistent')
        self._check_mark_list(mark_list)

        super().__init__(time_list=time_list,
                         obs_period=obs_period)
        self.mark_list = mark_list

    @staticmethod
    def _check_mark_list(mark_list: List[Any]):
        pass

    def append(self, time_stamp: float, mark: Any, obs_time_end=float('inf')):
        if time_stamp <= self.obs_period[1]:
            raise ValueError('the new time_stamp is inconsistent with obs_period.')
        if time_stamp > obs_time_end:
            raise ValueError('time_stamp and obs_time_end are not consistent.')
        if obs_time_end < self.obs_period[1]:
            raise ValueError('obs_time_end must be later than self.obs_period[1]')
        self.time_list.append(time_stamp)
        self.mark_list.append(mark)
        if obs_time_end == float('inf'):
            self.obs_period[1] = time_stamp
        else:
            self.obs_period[1] = obs_time_end
        return self

    def insert(self, time_stamp: float, mark: Any):
        insert_idx = bisect_left(self.time_list, time_stamp)
        if insert_idx == 0 and self.obs_period[0] > time_stamp:
            self.obs_period[0] = time_stamp
        elif insert_idx == len(self.time_list) and self.obs_period[1] < time_stamp:
            self.obs_period[1] = time_stamp
        else:
            pass
        self.time_list.insert(insert_idx, time_stamp)
        self.mark_list.insert(insert_idx, mark)
        return self

    @property
    def time_mark_list(self):
        return zip(self.time_list, self.mark_list)

    def batch_time_mark(self, time, len_window=float('inf')):
        rightmost_idx = bisect_left(self.time_list, time) - 1
        leftmost_idx = bisect_left(self.time_list, time - len_window)
        if rightmost_idx == -1:
            return [], []
        else:
            return self.time_list[leftmost_idx : rightmost_idx+1], \
                self.mark_list[leftmost_idx : rightmost_idx+1]

    def __add__(self, other):
        if self.obs_period != other.obs_period:
            raise ValueError('obs_period must be the same.')
        time_list = []
        mark_list = []
        self_idx = 0
        other_idx = 0
        for _ in range(len(self) + len(other)):
            if self.time_list[self_idx] < other.time_list[other_idx]:
                time_list.append(self.time_list[self_idx])
                mark_list.append(self.mark_list[self_idx])
                self_idx += 1
                if self_idx == len(self):
                    time_list = time_list + other.time_list[other_idx:]
                    mark_list = mark_list + other.mark_list[other_idx:]
                    break
            else:
                time_list.append(other.time_list[other_idx])
                mark_list.append(other.mark_list[other_idx])
                other_idx += 1
                if other_idx == len(other):
                    time_list = time_list + self.time_list[self_idx:]
                    mark_list = mark_list + self.mark_list[self_idx:]
                    break
        return self.__class__(time_list=time_list,
                              mark_list=mark_list,
                              obs_period=self.obs_period)

    def sample(self, size=1):
        idx_array = self.rng.choice(len(self), size=size, replace=True)
        return [(self.time_list[each_idx], self.mark_list[each_idx])\
                for each_idx in idx_array]


class MultivariateEventSeq(MarkedEventSeq):

    '''
    Data class for multivariate point process, represented as marked point process
    where a mark is a one-hot vector
    '''

    def __init__(self,
                 time_list: List[float],
                 mark_list: List[Any],
                 obs_period: List[float]):
        if len(time_list) != len(mark_list):
            raise ValueError('time_list and mark_list are inconsistent')
        self._check_mark_list(mark_list)

        super().__init__(time_list=time_list,
                         mark_list=mark_list,
                         obs_period=obs_period)
        self.mark_list = mark_list
        if self.mark_list:
            self.mark_tensor = torch.stack(self.mark_list)
        else:
            self.mark_tensor = torch.tensor([])

    @staticmethod
    def _check_mark_list(mark_list: List[torch.Tensor]):
        if not (all(is_n_d_tensor(each_mark, 1) for each_mark in mark_list)\
                and all(is_onehot(each_mark) for each_mark in mark_list)):
            raise ValueError('mark must be a list of one-hot 1d torch tensors.')

    @property
    def dim(self):
        return self.mark_list[0].numel()

    '''
    @property
    def mark_tensor(self):
        if not hasattr(self, '_mark_tensor'):
            self._mark_tensor = torch.stack(self.mark_list)
        elif len(self) != len(self._mark_tensor):
            self._mark_tensor = torch.stack(self.mark_list)
        else:
            pass
        return self._mark_tensor
    '''

    def append(self, time_stamp: float, mark: torch.Tensor, obs_time_end=float('inf')):
        if time_stamp <= self.obs_period[1]:
            raise ValueError('the new time_stamp is inconsistent with obs_period.')
        if time_stamp > obs_time_end:
            raise ValueError('time_stamp and obs_time_end are not consistent.')
        if obs_time_end < self.obs_period[1]:
            raise ValueError('obs_time_end must be later than self.obs_period[1]')
        self.time_list.append(time_stamp)
        self.mark_list.append(mark)
        self.mark_tensor = torch.cat((self.mark_tensor, mark.unsqueeze(0)))
        if obs_time_end == float('inf'):
            self.obs_period[1] = time_stamp
        else:
            self.obs_period[1] = obs_time_end
        return self

    def insert(self, time_stamp: float, mark: Any):
        insert_idx = bisect_left(self.time_list, time_stamp)
        if insert_idx == 0 and self.obs_period[0] > time_stamp:
            self.obs_period[0] = time_stamp
        elif insert_idx == len(self.time_list) and self.obs_period[1] < time_stamp:
            self.obs_period[1] = time_stamp
        else:
            pass
        self.time_list.insert(insert_idx, time_stamp)
        self.mark_list.insert(insert_idx, mark)
        self.mark_tensor = torch.cat((self.mark_tensor[:insert_idx],
                                      mark.unsqueeze(0),
                                      self.mark_tensor[insert_idx:]))
        #assert torch.allclose(torch.stack(self.mark_list), self.mark_tensor)
        return self

    def batch_time_mark_tensor(self, time, len_window=float('inf')):
        rightmost_idx = bisect_left(self.time_list, time) - 1
        leftmost_idx = bisect_left(self.time_list, time - len_window)
        if rightmost_idx == -1:
            return [], []
        else:
            return self.time_tensor[leftmost_idx : rightmost_idx+1], \
                self.mark_tensor[leftmost_idx : rightmost_idx+1]


class MultivariateDiffEventSeq(MultivariateEventSeq):

    '''
    Data class for differentiable multivariate point process represented as marked point process
    where a mark is a simplex vector.
    '''

    def __init__(self,
                 time_list: List[float],
                 mark_list: List[Any],
                 n_obs_dim: int,
                 n_hidden_dim: int,
                 obs_period: List[float]):
        super().__init__(time_list=time_list,
                         mark_list=mark_list,
                         obs_period=obs_period)
        if mark_list:
            self._check_dim(mark_list, n_obs_dim, n_hidden_dim)
        self.n_obs_dim = n_obs_dim
        self.n_hidden_dim = n_hidden_dim
        self.hidden_time_list = []
        self.hidden_mark_list = []

    @staticmethod
    def _check_mark_list(mark_list: List[torch.Tensor]):
        if not ((all(is_n_d_tensor(each_mark, 1) for each_mark in mark_list) \
                 and all(is_simplex(each_mark) for each_mark in mark_list))):
            raise ValueError('each mark must be within a probability simplex.')

    def _check_dim(self, mark_list, n_obs_dim, n_hidden_dim):
        if (not all(self.mark_tensor[:, :n_obs_dim].sum(axis=1) \
                   == torch.ones(len(mark_list)))) \
                   and self.mark_tensor.shape[1] == n_obs_dim + n_hidden_dim\
                   and (not all(self.mark_tensor[:, n_obs_dim:].sum(axis=1) \
                                == torch.zeros(len(mark_list)))):
            raise ValueError('mark_list is inconsistent with n_obs_dim and/or n_hidden_dim.')

    def insert_hidden(self, time_stamp: float, mark: torch.Tensor):
        self.insert(time_stamp, mark)

        # insert to hidden lists
        insert_idx = bisect_left(self.hidden_time_list, time_stamp)
        if insert_idx == 0 and self.obs_period[0] > time_stamp:
            raise ValueError
        elif insert_idx == len(self.time_list) and self.obs_period[1] < time_stamp:
            raise ValueError
        else:
            pass
        self.hidden_time_list.insert(insert_idx, time_stamp)
        self.hidden_mark_list.insert(insert_idx, mark)
        return self

    @property
    def hidden_time_mark_list(self):
        return zip(self.hidden_time_list, self.hidden_mark_list)

    def sample_hidden(self, size=1):
        if self.hidden_time_list:
            idx_array = self.rng.choice(len(self.hidden_time_list), size=size, replace=True)
        else:
            idx_array = []
        return [(self.hidden_time_list[each_idx], self.hidden_mark_list[each_idx])\
                for each_idx in idx_array]


class MultivariateLogDiffEventSeq(MultivariateDiffEventSeq):

    ''' the last dimension represents null spikes
    '''

    @staticmethod
    def _check_mark_list(mark_list: List[torch.Tensor]):
        if not (all(is_n_d_tensor(each_mark, 1) for each_mark in mark_list)):
            raise ValueError('each mark must be within a probability simplex.')

    def _check_dim(self, mark_list, n_obs_dim, n_hidden_dim):
        if self.mark_tensor.shape[1] != n_obs_dim + n_hidden_dim + 1:
            raise ValueError('mark_list is inconsistent with n_obs_dim and/or n_hidden_dim.')

    def batch_time_mark_tensor(self, time, len_window=float('inf')):
        rightmost_idx = bisect_left(self.time_list, time) - 1
        leftmost_idx = bisect_left(self.time_list, time - len_window)
        if rightmost_idx == -1:
            return [], []
        else:
            return self.time_tensor[leftmost_idx : rightmost_idx+1], \
                torch.exp(self.mark_tensor[leftmost_idx : rightmost_idx+1, :-1])
        # drop the last dim, which is the auxiliary variable.
        # apply exp because all the marks are in the log domain


def append_diff_hidden(history: MultivariateEventSeq,
                       n_hidden_neurons: int)->MultivariateDiffEventSeq:
    ''' append hidden dimensions

    Parameters
    ----------
    history : MultivariateEventSeq
        each mark is a tensor whose shape is (n_obs_neurons,)

    Returns
    -------
    history : MultivariateDiffEventSeq
        each mark is a tensor whose shape is (n_obs_neurons+n_hidden_neurons,)
    '''
    return MultivariateLogDiffEventSeq(
        time_list=deepcopy(history.time_list),
        mark_list=list(torch.unbind(
            torch.log(torch.cat((history.mark_tensor,
                                 torch.zeros(len(history), n_hidden_neurons + 1)),
                                dim=1)))),
        n_obs_dim=history.dim,
        n_hidden_dim=n_hidden_neurons,
        obs_period=history.obs_period)


def append_hidden(history: MultivariateEventSeq,
                  n_hidden_neurons: int)->MultivariateDiffEventSeq:
    ''' append hidden dimensions

    Parameters
    ----------
    history : MultivariateEventSeq
        each mark is a tensor whose shape is (n_obs_neurons,)

    Returns
    -------
    history : MultivariateDiffEventSeq
        each mark is a tensor whose shape is (n_obs_neurons+n_hidden_neurons,)
    '''
    return MultivariateDiffEventSeq(
        time_list=deepcopy(history.time_list),
        mark_list=list(torch.unbind(
            torch.cat((history.mark_tensor,
                       torch.zeros(len(history),
                                   n_hidden_neurons)),
                                dim=1))),
        n_obs_dim=history.dim,
        n_hidden_dim=n_hidden_neurons,
        obs_period=history.obs_period)


def delete_hidden(history: MultivariateEventSeq,
                  n_obs_neurons: int) -> MultivariateEventSeq:
    ''' delete hidden dimensions

    Parameters
    ----------
    history : MultivariateEventSeq
        each mark is a tensor whose shape is (n_obs_neurons,)

    Returns
    -------
    history : MultivariateDiffEventSeq
        each mark is a tensor whose shape is (n_obs_neurons+n_hidden_neurons,)
    '''
    mark_tensor = history.mark_tensor[:, :n_obs_neurons]
    nonzero_list = mark_tensor.sum(dim=1).nonzero(as_tuple=False).reshape(-1)
    return MultivariateEventSeq(
        time_list=torch.tensor(history.time_list)[nonzero_list].tolist(),
        mark_list=list(torch.unbind(mark_tensor[nonzero_list])),
        obs_period=history.obs_period)
