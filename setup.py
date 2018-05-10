"""Install script."""

from setuptools import find_packages, setup


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

setup(
    name="signal_utils",
    description='',
    version="0.1.22",
    author="Vasiliy Chernov",
    packages=find_packages(),
    platforms='any',
    install_requires=parse_requirements("signal_utils/requirements.txt"),
    include_package_data=True,
    package_data={
        '': ['*.h5', '*.dat'],
    }
)
