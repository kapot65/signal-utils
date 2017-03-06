# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 16:02:13 2017

@author: kapot
"""

from setuptools import setup
from pip.req import parse_requirements

install_reqs = parse_requirements("signal_utils/requirements.txt", 
                                  session='hack')
reqs = [str(ir.req) for ir in install_reqs]

setup(
    name = "signal_utils",
    description='',
    version = "0.1.4",
    author = "Vasiliy Chernov",
    packages = ["signal_utils"],
    platforms = 'any',
    install_requires = reqs,
)