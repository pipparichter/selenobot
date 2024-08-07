import setuptools
import os

modules = ['selenobot.' + f.replace('.py', '') for f in os.listdir('./selenobot')]
scripts = ['selenobot.' + f.replace('.py', '') for f in os.listdir('./scripts')]

setuptools.setup(
    name='selenobot',
    version='0.1',    
    description='A protein language model-based framework which uses protein language models to detect erroneous truncation of selenoproteins',
    url='https://github.com/pipparichter/selenobot',
    author='Philippa Richter',
    author_email='prichter@caltech.edu',
    packages=['selenobot'],
    install_requires=setuptools.find_packages(exclude=['selenobot'] + modules + scripts))
