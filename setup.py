from setuptools import setup, find_packages
from os.path import join, dirname
import risksutils


setup(
    name='risksutils',
    version=risksutils.__version__,
    packages=find_packages(),
    long_description=open(join(dirname(__file__), 'README.rst')).read(),
    install_requires=[
        'pandas>=0.20.3',
        'holoviews>=1.8.3',
        'numpy>=1.13.1',
        'scikit-learn>=0.19.0',
        'scipy>=0.19.0',
    ]
)
