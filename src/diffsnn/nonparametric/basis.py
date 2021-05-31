#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = 'Copyright IBM Corp. 2020, 2021'

import torch


class Uniform(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.support = (-1., 1.)

    def forward(self, u):
        return (-1. <= u) * (u <= 1.) * 0.5


class Triangle(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.support = (-1., 1.)

    def forward(self, u):
        return (-1. <= u) * (u <= 1.) * (1.0 - torch.abs(u))


class Epanechnikov(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.support = (-1., 1.)

    def forward(self, u):
        return (-1. <= u) * (u <= 1.) * (1.0 - u ** 2) * 0.75


class Gaussian(torch.nn.Module):

    def __init__(self):
        super.__init__()
        self.support = (-float('inf'), float('inf'))

    def forward(self, u):
        return (1.0 / torch.sqrt(2.0 * torch.pi)) * torch.exp(-0.5 * (u ** 2))
