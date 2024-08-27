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
Download the Sharp dataset called [train_sharp](https://drive.google.com/open?id=1YLksKtMhd2mWyVSkvhDaDLWSc1qYNCz-) and Low Resolution dataset called [train_sharp_bicubic](https://drive.google.com/open?id=1a4PrjqT-hShvY9IyJm3sPF0ZaXyrCozR) from [REDS dataset](https://seungjunnah.github.io/Datasets/reds) to the `datasets` directory at the same level of file `compile_dataset.ipynb`.

Once downloaded, the images should convert them to gray-scale by running *compile_dataset.ipynb*.

#### Model fine tuning
Fine tuning data are not available at this moment. A Pytorch model fine-tuned on high-speed imaging data can be found in the `model` directory.

### Training
Change directory to `train` and run
```
python train_reds_gray.py
```

### Inference
Change directory to `inference` and run
```
python infer.py
```