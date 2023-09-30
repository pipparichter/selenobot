from setuptools import find_packages, setup

setup(
	name = "selenobot",
	version = '0.0',
	description = 'Python package for using protein language models to predict selenoproteins',
	url = 'https://github.com/prichter/selenobot',
	author = 'Pippa Ritchter, Joshua E. Goldford',
	author_email = 'prichter@caltech.edu, goldford@caltech.edu',
	packages = find_packages(),
	install_requires = [],
	include_package_data = True,
)
