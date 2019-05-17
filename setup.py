# -*- coding: utf-8 -*-

from os.path import join, dirname
from setuptools import setup
import risksutils


setup(
    name='risksutils',
    version=risksutils.__version__,
    packages=[
        'risksutils'
    ],
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='woe credit scoring information-value visualization',
    description='Scripts for credit scoring analysis',
    long_description=open(join(dirname(__file__), 'README.rst')).read(),
    url='https://github.com/dshulchevskii/risksutils',
    author='Dmitry Shulchevskii',
    author_email='dshulchevskii@gmail.com',
    install_requires=[
        'pandas>=0.22.0',
        'holoviews>=1.9.0',
        'numpy>=1.13.1',
        'scikit-learn>=0.19.0',
        'scipy>=0.19.1',
        'matplotlib>=2.0.2',
        'bokeh>=0.12.10',
        'statsmodels>=0.8.0',
    ],
    setup_requires=[
        "pytest-runner",
        "pytest-pylint",
        # "pytest-pycodestyle",
        "pytest-pep257",
        "pytest-cov",
    ],
    tests_require=[
        "pytest",
        "pylint",
        # "pycodestyle",
        "pep257",
    ],
)
