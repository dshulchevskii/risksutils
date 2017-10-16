from os.path import join, dirname
from setuptools import setup
import risksutils


setup(
    name='risksutils',
    version=risksutils.__version__,
    packages=[
        'risksutils',
        'risksutils/visualization',
        'risksutils/metrics',
    ],
    long_description=open(join(dirname(__file__), 'README.md')).read(),
    install_requires=[
        'pandas>=0.20.3',
        'holoviews>=1.8.3',
        'numpy>=1.13.1',
        'scikit-learn>=0.19.0',
        'scipy>=0.19.0',
        'matplotlib>=2.0.2',
        'bokeh>=0.12.6',
    ],
    setup_requires=[
        "pytest-runner",
        "pytest-pylint",
        "pytest-pycodestyle",
        "pytest-pep257",
        "pytest-cov",
    ],
    tests_require=[
        "pytest",
        "pylint",
        "pycodestyle",
        "pep257",
    ],
)
