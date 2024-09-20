import setuptools
import os

def get_requirements(path:str=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'requirements.txt')):
    with open(path) as f:
        requirements = f.read().splitlines()
    return requirements

setuptools.setup(
    name='selenobot',
    version='0.1',    
    description='A protein language model-based framework which uses protein language models to detect erroneous truncation of selenoproteins',
    url='https://github.com/pipparichter/selenobot',
    author='Philippa Richter',
    author_email='prichter@caltech.edu',
    packages=['selenobot'], 
    install_requires=get_requirements())
