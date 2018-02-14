"""Install script."""

from pip.req import parse_requirements
from setuptools import find_packages, setup

INSTALL_REQS = parse_requirements("signal_utils/requirements.txt",
                                  session='hack')
REQS = [str(ir.req) for ir in INSTALL_REQS]

setup(
    name="signal_utils",
    description='',
    version="0.1.21",
    author="Vasiliy Chernov",
    packages=find_packages(),
    platforms='any',
    install_requires=REQS,
    include_package_data=True,
    package_data={
        '': ['*.h5', '*.dat'],
    }
)
