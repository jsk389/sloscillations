from setuptools import setup, find_packages

setup(
    name='sloscillations',
    version='0.1.dev0',
    license='LICENSE.txt',
    description='A package for generating artificial solar-like oscillations',
    #long_description=open('README.txt').read(),
    packages=['sloscillations', 'sloscillations.tests'],
)