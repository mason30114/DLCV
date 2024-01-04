# DLCV HW3 

# Install required packages
* Please install required packages as the following shell script:
```shell script=
conda create --name hw3 python=3.8
conda activate hw3
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
bash hw3_download.sh
```

# Inference
* Please run the inference code as the following shell script: <br>
* Problem 1:
```shell script=
bash hw3_1.sh <Path to folder containing test images> <Path to the id2label.json> <Path to output csv file>
```
* Problem 2:
```shell script=
bash hw3_2.sh <Path to folder containing test images> <Path to output json file> <Path to the decoder weights>
```
