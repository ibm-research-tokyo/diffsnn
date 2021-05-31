#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = 'Copyright IBM Corp. 2020, 2021'

import unittest
import torch
from diffsnn.utils import idx2onehot, onehot2idx, ExpConcreteDistribution

class TestUtils(unittest.TestCase):

    def test_onehot(self):
        n_categories = 5
        sample_size = 10
        idx_tensor = torch.LongTensor(sample_size, 1).random_() % n_categories
        onehot = idx2onehot(idx_tensor, n_categories)
        for each_data_idx in range(sample_size):
            for each_cat_idx in range(n_categories):
                if each_cat_idx == idx_tensor[each_data_idx]:
                    self.assertEqual(onehot[each_data_idx, each_cat_idx], 1.0)
                else:
                    self.assertEqual(onehot[each_data_idx, each_cat_idx], 0.)
        idx_tensor_recovered = onehot2idx(onehot)
        self.assertTrue(all(idx_tensor.reshape(-1) == idx_tensor_recovered.reshape(-1)))

    def test_concrete(self):
        dist = ExpConcreteDistribution(4)
        print(dist.log_pdf(torch.log(torch.tensor([0.99997, 0.00001, 0.00001, 0.00001])),
                           torch.tensor(1e-8),
                           logits=torch.tensor([10.0, 0.0, 0.0, 0.0])))
