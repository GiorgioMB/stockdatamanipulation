"""This module sets up the package for distribution."""
from setuptools import setup, find_packages

setup(
    name='stockdatafetcher',
    version='0.1',
    packages=find_packages(),
    description='A comprehensive data retrieval aimed at gathering financial information',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Giorgio Micaletto',
    author_email='giorgio.micaletto@studbocconi.it',
    url='https://github.com/GiorgioMB/stockdatafetcher/',
    install_requires=[
        'pandas>=1.1.5',
        'yfinance>=0.1.63',
        'requests>=2.25.1'
    ],
    python_requires='>=3.6',
)
