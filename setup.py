# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 16:02:13 2017

@author: kapot
"""

from pip.req import parse_requirements
from setuptools import find_packages
from setuptools import setup

INSTALL_REQS = parse_requirements("signal_utils/requirements.txt",
                                  session='hack')
REQS = [str(ir.req) for ir in INSTALL_REQS]

setup(
    name="signal_utils",
    description='',
    version="0.1.14",
    author="Vasiliy Chernov",
    packages=find_packages(),
    platforms='any',
    install_requires=REQS,
    dependency_links=["https://github.com/kapot65/random_custom_pdf/archive/master.zip"],
    include_package_data=True,
    package_data={
        '': ['*.h5', '*.dat'],
    }
)
