#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = 'Copyright IBM Corp. 2020, 2021'

from torch.distributions.relaxed_categorical import ExpRelaxedCategorical
import torch

def idx2onehot(idx_tensor: torch.Tensor, n_categories: int) -> torch.FloatTensor:
    ''' Convert a tensor containing indices into one-hot vectors

    Parameters
    ----------
    idx_tensor : torch.LongTensor, shape (sample_size,) or (sample_size, 1)
    n_categories : int
    '''
    if not (all(idx_tensor <= n_categories - 1) and all(idx_tensor >= 0)):
        raise ValueError('idx_tensor and n_categories are not consistent.')
    sample_size = idx_tensor.shape[0]
    onehot = torch.FloatTensor(sample_size, n_categories)
    onehot.zero_()
    onehot.scatter_(1, idx_tensor, 1)
    return onehot

def onehot2idx(onehot_tensor: torch.FloatTensor) -> torch.LongTensor:
    ''' Inverse of `idx2onehot`

    Parameters
    ----------
    onehot_tensor : torch.FloatTensor
    '''
    nonzero_idx = onehot_tensor.nonzero(as_tuple=False)
    if not all(nonzero_idx[:, 0] == torch.arange(onehot_tensor.shape[0])):
        raise ValueError('onehot_tensor is not *one-hot*')
    return nonzero_idx[:, 1]

def log1mexp(input_tensor):
    ''' compute log (1 - exp(-|input|)) accurately.
    '''
    abs_input = torch.abs(input_tensor)
    mask = abs_input < 0.6931471805599453 # log 2
    return torch.log(-torch.expm1(-abs_input)) * mask \
        + torch.log1p(-torch.exp(-abs_input)) * (~mask)

def complete_logprob(log_sub_prob):
    if len(log_sub_prob) == 1:
        return torch.cat(
            (log_sub_prob,
             log1mexp(log_sub_prob).reshape(-1)))
    else:
        logsumprob = torch.logsumexp(log_sub_prob, dim=0)
        if logsumprob >= 0:
            raise ValueError('sum of probs must be smaller than 1.')
        return torch.cat(
            (log_sub_prob,
             log1mexp(logsumprob).reshape(-1)))


class ExpConcreteDistribution:

    ''' distribution on the logarithm of simplex vectors
    '''

    def __init__(self, dim):
        self.dist = ExpRelaxedCategorical(torch.tensor(1.0),
                                          logits=torch.zeros(dim))

    def set_params(self, temperature, logits):
        self.dist.temperature = temperature
        self.dist._categorical.logits = logits

    def log_pdf(self, log_p, temperature, logits):
        ''' compute log pdf

        Parameters
        ----------
        log_p : torch.tensor
            logarithm of a simplex vector
        '''
        self.set_params(temperature, logits)
        return self.dist.log_prob(log_p)

    def pdf(self, log_p, temperature, logits):
        ''' compute pdf of the realization `log_p`
        '''
        return torch.exp(self.log_pdf(log_p, temperature, logits))

    def rsample(self, temperature, logits, sample_shape=torch.Size()):
        ''' random sampling
        '''
        self.set_params(temperature, logits)
        return self.dist.rsample(sample_shape)
