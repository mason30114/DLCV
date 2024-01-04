# DLCV HW1 

# Install required packages
* Please install required package as the following shell script:
```shell script=
conda create --name hw1 python=3.8
conda activate hw1
pip install -r 'requirements.txt'
```

# Install Dataset
* Please dataset as the following shell script:
```shell script=
bash get_dataset.sh
```

# Install checkpoints
* Please install required checkpoints as the following shell script:
```shell script=
bash hw1_download_ckpt.sh
```

# Inference
* Please run the inference code as the following shell script: <br>
* Problem 1:
```shell script=
bash hw1_1.sh <Path to testing images directory> <Path to output csv file> 
```
* Problem 2:
```shell script=
bash hw1_2.sh <Path to images csv file> <Path to folder containing images> <Path to output csv file>
```
* Problem 3:
```shell script=
bash hw1_3.sh <Path to testing images directory> <Path to output images directory> 
```

