from setuptools import setup, find_packages

setup(
    name='selenobot',
    # version='0.1',    
    description='Tool which uses protein language models to detect erroneous truncation of selenoproteins',
    url='https://github.com/pipparichter/selenobot',
    author='Philippa Richter',
    author_email='prichter@caltech.edu',
    packages=['src'],
    install_requires=find_packages(exclude=['src']))

