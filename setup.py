import setuptools
from selenobot.setup import setup

setuptools.setup(
    name='selenobot',
    # version='0.1',    
    description='Tool which uses protein language models to detect erroneous truncation of selenoproteins',
    url='https://github.com/pipparichter/selenobot',
    author='Philippa Richter',
    author_email='prichter@caltech.edu',
    packages=['selenobot'],
    install_requires=setuptools.find_packages(exclude=['selenobot', 'setup', 'setup.data', 'selenobot.plot']))


