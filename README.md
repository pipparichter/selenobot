# Selenobot

## Requirements
What are the packages and software the user is already assumed to have installed? So far, I think it is just Python and Miniconda (or Anaconda).

## Installation

First, create a new conda environment which runs the latest version of Python (3.11.5 at the time of writing). Then activate the environment. 
```
conda create -n selenobot python=3.11.5
conda activate selenobot
```
Clone the `selenobot` repository into the current working directory. 
```
git clone https://github.com/pipparichter/selenobot.git
```
Use `pip` to install the `selenobot` package into the `selenobot` conda environment. This should also install all Python dependencies into the environment.  
% Apparently the -e option just pts the package in "editable mode", which means that any changes made are reflected in the environment. I am not sure if this is actually 
% necessary when a user is installing it. 
```
pip install selenobot
```

## Accessing data

Curated training, validation, and testing data are stored in a publically-available Google Cloud bucket. The data can be accessed using the `download_training_data` function, defined in teh `selenobot.io` module. 
