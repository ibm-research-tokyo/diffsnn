#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" setup file. """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "Copyright IBM Corp. 2020, 2021"

from setuptools import setup, find_packages

def _requires_from_file(filename):
    return open(filename).read().splitlines()

setup(
    name='diffsnn',
    package_dir={"":"src"},
    packages=find_packages("src", exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    test_suite="tests",
    install_requires=_requires_from_file('requirements.txt'),
)
