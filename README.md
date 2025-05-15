# PINGS
This repository will contain the implementation of the paper 'PINGS: Gaussian Splatting Meets Distance Fields within a Point-Based Implicit Neural Map' ([paper link](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/pan2025rss.pdf))

**TL;DR: PINGS is a LiDAR-visual SLAM system unifying distance fields and radiance fields within a neural point map**

![image](https://github.com/user-attachments/assets/0ec8b71d-8902-445c-a07c-f9a7e08fb3e2)


🚧 **Repo under construction** 🚧


## Installation

### 1. Clone the repository

```
git clone git@github.com:PRBonn/PINGS.git --recursive
cd PINGS
```

### 2. Set up conda environment

```
conda create --name pings python=3.10
conda activate pings
```

### 3. Install the key requirement PyTorch

```
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

The commands depend on your CUDA version (check it by `nvcc --version`). You may check the instructions [here](https://pytorch.org/get-started/previous-versions/).


### 4. Install other dependency

```
pip3 install -r requirements.txt
```

## Run PINGS


To check how to run PINGS, you can use the following command:

```
python3 pings.py -h 
```

To check how to inspect the results afterwards, you can use the following command:

```
python3 inspect_pings.py -h 
```