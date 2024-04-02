# DLCV HW2

# Project Introduction
* Please see "2023_fall_hw2_intro.pptx.pdf" to understand the details.

# Install required packages
* Please install required packages as the following shell script:
```shell script=
conda create --name hw2 python=3.10
conda activate hw2
pip install -r 'requirements.txt'
```

# Install Dataset
* Please download the dataset as the following shell script:
```shell script=
bash get_dataset.sh
```

# Install checkpoints
* Please install required checkpoints as the following shell script:
```shell script=
bash hw2_download.sh
```

# Inference
* Please run the inference code as the following shell script: <br>
* Problem 1:
```shell script=
bash hw2_1.sh <Path to directory of generated images> 
```
* Problem 2:
```shell script=
bash hw2_2.sh <Path to directory of predefined noises> <Path to directory of generated images> <Path to the pretrained model weight>
```
* Problem 3:
```shell script=
bash hw2_3.sh <Path to testing images in the target domain> <Path to output prediction file> 
```
