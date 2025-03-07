<div align="center">
  <p>
        <img src="images/logo.png" width="220">
    </a>
</p>
</div>

# MToIE
Multi-Knowledge-oriented Nighttime Haze Imaging Enhancer for Vision-driven Intelligent Systems

# [[Paper](https://arxiv.org/abs/2502.07351)]
> **Abstract:** Salient object detection (SOD) plays a critical role in vision-driven measurement systems (VMS), facilitating the detection and segmentation of key visual elements in an image. However, adverse imaging conditions such as haze during the day, low light, and haze at night severely degrade image quality, and complicating the SOD process. To address these challenges, we propose a multi-task-oriented nighttime haze imaging enhancer (MToIE), which integrates three tasks: daytime dehazing, low-light enhancement, and nighttime dehazing. The MToIE incorporates two key innovative components: First, the network employs a task-oriented node learning mechanism to handle three specific degradation types: day-time haze, low light, and night-time haze conditions, with an embedded self-attention module enhancing its performance in nighttime imaging. In addition, multi-receptive field enhancement module that efficiently extracts multi-scale features through three parallel depthwise separable convolution branches with different dilation rates, capturing comprehensive spatial information with minimal computational overhead. To ensure optimal image reconstruction quality and visual characteristics, we suggest a hybrid loss function. Extensive experiments on different types of weather/imaging conditions illustrate that MToIE surpasses existing methods, significantly enhancing the accuracy and reliability of vision systems across diverse imaging scenarios.


# 📂 Datasets
The training and testing datasets include realistic single image dehazing (RESIDE) OTS and the composite degradation dataset (CDD).

# 📄 Checkpoint
checkpoint will be released later! Or you can train it by yourself.

# ⚙️ Installation  
This codebase was tested with the following environment configurations:

- Ubuntu 20.04
- CUDA 11.8
- Python 3.8
- PyTorch 1.11.0 + cu113

# 🔥 Training  
1. Please download the corresponding training datasets and put them in the folder.
2. Please run the `prepare_patches.py` and check the `Train.h5` file.
3. Begin training our model.
```
python Train.py
```

# 🔥 Testing
1. Please download the corresponding testing datasets and put them in the other folder.
2. Please check the `checkpoint.pth.tar` file.
3. Begin testing our model.
```
python Test.py
```

# Citation  
