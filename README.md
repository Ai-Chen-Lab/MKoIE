<div align="center">
  <p>
        <img src="images/logo.png" width="100">
    </a>
</p>

# MToIE
Multi-Task-oriented Nighttime Haze Imaging Enhancer for Vision-driven Measurement Systems

# [[Paper]]
> **Abstract:** 


# Datasets
The training and testing datasets include realistic single image dehazing (RESIDE) OTS and the composite degradation dataset (CDD).

# Checkpoint
checkpoint will be released later! Or you can train it by yourself.

# Installation  
```
pip install -r requirements.txt
```

# Training  
1. Please download the corresponding training datasets and put them in the folder.
2. Please run the prepare_patches.py and check the `Train.h5` file.
3. Begin training our model.
```
python Train.py
```

# Testing
1. Please download the corresponding testing datasets and put them in the other folder.
2. Please check the `checkpoint.pth.tar` file.
3. Begin testing our model.
```
python Test.py
```

# Citation  
