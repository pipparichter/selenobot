<p align="center">
<img src="./mascot.png" width="400" height="400">
</p>

# Selenobot

`selenobot` is a computational framework which uses transformer-generated embeddings of protein sequences to detect misannotated selenoproteins in genome databases. This repository contains code for reproducing the training of the classifiers used for selenoprotein prediction, as well as basic plotting utilities for visualizing the results.

This repository contains code which handles the selenoprotein "detection" task. The classifiers associated with this project only flag potentially-misannotated selenoproteins. Additional functionality for this tool is in development, which will enable a user to extend the selenoprotein beyond the first selenocysteine residue to the true STOP codon. Evnetually, the detection and extension frameworks will be unified as a multipurpose software tool. 


## Requirements

Prior to installing `selenobot`, you must have a version of Python, as well as Miniconda or Anaconda to manage the Python environment. For performing homology clustering on the sequence data, you must also have the CD-HIT tool installed. This software can be downloaded at [this link](https://sites.google.com/view/cd-hit). 

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
cd selenobot # Important so that the directory structure is properly set up. 
pip install . 
```

Once the package has been installed, you must set up the datasets for classifier training, testing, and validation. The scripts for doing this are contained in the `setup` subdirectory. To initiate setup, simply run the following lines in the terminal. These lines download the training data from a [Google Cloud bucket]('https://storage.googleapis.com/selenobot-data/'), and organize the data directory structure. It also sets up the `selenobot.cfg` file, which contains the paths to locally-stored data files, as well as settings for the CD-HIT clustering program. 

```
cd setup
python main.py DATA_DIR --cdhit CDHIT
```

`DATA_DIR` is an absolute path specifying the location where the data will be stored. 
`CDHIT` is the absolute path to the CD-HIT command. If unspecified, it is assumed that the program is installed in the user's home directory.

## Usage


