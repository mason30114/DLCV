# DLCV Final Project ( Visual Queries 2D Localization Task )

# Project Introduction
* Please see the part of challenge 2 in "DLCV Fall 2023 - Final Project.pdf" to understand the details.

# Reference
* This project is built on [VQLoC](https://github.com/hwjiang1510/VQLoC):

# Install required packages
* Please install required package as the following shell script:
```shell script=
conda create --name lolidyulunloveyuki python=3.8
conda activate lolidyulunloveyuki

# Install newest pytorch 
pip install torch torchvision torchaudio
pip install -r 'requirements.txt'
```

# Install images for training and validation
* Please install required images as the following shell script:
* If download fail, please add this [dataset](https://drive.google.com/uc?id=1JjVYJnRlfRvvy6195OSISr0nAK_mAmbD) on the root dir, and directly unzip the image.zip as ./images/
```shell script=
bash download_images.sh
```
# Install checkpoints
* Please install required checkpoints as the following shell script:
```shell script=
bash download_ckpt.sh
```

# Training
* Please run the training code as the following shell script:
```shell script=
bash train.sh <Path to clips folder> <Path to annot file folder>
```
"Path to clips folder" contains all the clips for training, validation and testing.<br>
"Path to annot file folder" contains original training and validation annot json file (vq_train.json, vq_val.json). Our default setting is './orig_anno/'.<br>
In train.sh, you can replace 'False' by 'True' after each augmentation method to activate different training strategy. For example: --Use_UFS_3 'True'.<br>

# Inference
* Please run the inference code as the following shell script:
```shell script=
bash inference.sh <Path to clips folder> <Path to unannot file> <Path to output json file>
```
"Path to clips folder" contains all the clips for training, validation and testing.<br>
"Path to unannot file" is the path to unannotated testing json file.<br>
"Path to output json file" is the path to prediction json file.<br>
* Please ensure that the folder of output exists.

# Visualization
* Please run the visualization visualization as the following shell script:
```shell script=
bash visualize_annotations.sh <Path to unannot file> <Path to prediction file> <Path to clips folder>
```
"Path to unannot file" is the path to unannotated testing json file.<br>
"Path to prediction file" is the path to prediction json file (the output of inference step).<br>
"Path to clips folder" contains all the clips for training, validation and testing.<br>
This step will give you videos with generated bounding-box in './videos' folder.<br>
* You may need to change the environment to run visualization code. 
* Please refer to the "Basic installation" step of [this repo](https://github.com/facebookresearch/vq2d_cvpr/blob/main/INSTALL.md)


