"""This module sets up the package for distribution."""
from setuptools import setup, find_packages

setup(
    name='stockdatamanager',
    version='0.10',
    packages=find_packages(),
    description='A comprehensive library for financial analysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Giorgio Micaletto',
    author_email='giorgio.micaletto@studbocconi.it',
    url='https://github.com/GiorgioMB/stockdatafetcher/',
    install_requires=[
        'pandas>=1.1.5',
        'yfinance>=0.1.63',
        'requests>=2.25.1',
        'pylint>=2.6.0',
        'pandas-datareader>=0.10.0',
        'numpy>=1.23',
        'scipy>=1.7.1',
        'matplotlib>=3.4.2',
        'seaborn>=0.11.1',
        'arch>=4.19',
        'optuna>=2.10.0'
    ],
    python_requires='>=3.6',
)
