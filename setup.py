"""This module sets up the package for distribution."""
from setuptools import setup, find_packages

setup(
    name='datafetcher',
    version='0.1',
    packages=find_packages(),
    description='A comprehensive data retrieval aimed at gathering financial information',
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description=f.read()    
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
