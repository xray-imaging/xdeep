# XFusion
Deep learning-based spatiotemporal fusion for high-fidelity ultra-high-speed x-ray radiography  
A model to reconstruct high quality x-ray images by combining the high spatial resolution of high-speed camera and high temporal resolution of ultra-high-speed camera image sequences.  

## Prerequisites
This implementation is based on the [BasicSR toolbox](https://github.com/XPixelGroup/BasicSR). Data for model pre-training are collected from the [REDS dataset](https://seungjunnah.github.io/Datasets/reds).  

## Usage
### Virtual environment
Conda virtual environment (venv) can be installed following `conda env create -f /path/to/stf_env_conda.yml`.  
This venv has been tested on Python 3.11. Once installed, activate the venv following `conda activate stf_env`.

### Data preparation
#### Model pretraining
Download the high resolution and low resolution image data from [REDS dataset](https://seungjunnah.github.io/Datasets/reds) to the datasets directory at the same level of the file 
Once downloaded, the images can be converted to gray-scale by running 
