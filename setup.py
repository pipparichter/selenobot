import setuptools

setuptools.setup(
    name='selenobot',
    version='0.1',    
    description='A protein language model-based framework which uses protein language models to detect erroneous truncation of selenoproteins',
    url='https://github.com/pipparichter/selenobot',
    author='Philippa Richter',
    author_email='prichter@caltech.edu',
    packages=['selenobot', 'selenobot.plot'],
    install_requires=setuptools.find_packages(exclude=['selenobot', 'selenobot.plot']))


