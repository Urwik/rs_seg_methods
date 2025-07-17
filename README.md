# Reticular Structures Segmentation Methods

<p align="center">
    <img src="docs/gazebo.gif" alt="Gazebo Environment Example" width="45%" style="display:inline-block; margin-right:10px;"/>
    <img src="docs/rviz.gif" alt="Segmentation Example Animation" width="45%" style="display:inline-block;"/>
</p>

<!-- <p align="center">
    <img src="docs/Fig9_dl_crossed_00226_ptv3.png" alt="Deep Learning Segmentation Example" width="90%" style="object-fit:cover; object-position:center; height:180px;"/>
</p> -->

---

## Overview

This repository contains the source code for the segmentation of reticular structures using different methods, both analytical and deep learning-based. Data is obtained using a simulated 3D LiDAR sensor. All implemented methods aim to perform binary segmentation of the structures from the background.

## Table of Contents
- [Requirements](#requirements)
- [Build](#build)
- [Usage](#usage)
- [Analytical](#analytical)
- [Deep Learning](#deep-learning)
- [Results](#results)
- [Credits](#credits)
- [Citation](#citation)

## Requirements

- Ubuntu 20.04 or later 
- ROS Noetic (for ROS-based modules)
- Python 3.8+
- C++17 compatible compiler (for analytical modules)
- [PointNet and PointNet++ (PyTorch implementation)](https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git)
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) (for MinkUNet34C)
- [Pointcept](https://github.com/Pointcept/Pointcept) (for PointTransformerV3)
- Additional Python packages: numpy, torch, pyyaml, pandas, matplotlib, etc.


## Build
### Analytical Methods, Gazebo Generator Plugin, Truss Generator (C++/ROS)
To build the analytical modules, navigate to your catkin workspace (e.g., `~/catkin_ws`) and run:

```bash
cd ~/catkin_ws
catkin_make
```

### Deep Learning Methods (Python)
No build step is required, but ensure all dependencies are installed. You can install the required Python packages using:

```bash
pip install -r deep_learning/requirements.txt
```

## Usage

### Analytical Segmentation
Run the analytical segmentation node (from your catkin workspace):
```bash
rosrun analytical_rs_seg node
```

### Deep Learning Segmentation
Train a model:
```bash
python3 deep_learning/scripts/train.py 
```
Test a model:
```bash
python3 deep_learning/scripts/test.py 
```
#### PointTransformerV3 (Pointcept)

Train the PointTransformerV3 model:
```bash
cd deep_learning/repos/Pointcept
python3 tools/train_retTruss.py
```
Test the PointTransformerV3 model:
```bash
python3 tools/test_retTruss.py
```

### Truss Generation 
To generate reticular structure models, run:
```bash
rosrun truss_generator arvc_build_truss
```

### Data Generation (Gazebo)
Launch Gazebo world and plugin for data generation:
```bash
roslaunch gazebo_generator_plugin example_train.launch
```

## Repository Structure
The repository is organised into the following main directories:
- `analytical_rs_seg`: Analytical methods for segmentation.
- `deep_learning`: Deep learning methods for segmentation.
- `gazebo_generator_plugin`: Synthetic data generation in Gazebo.
- `truss_generator`: Generation of reticular structure models.

## Analytical
This module implements an algorithm to segment reticular structures in outdoor environments. The algorithm requires prior knowledge of the structure, such as the length and width of its bars. The workflow of the algorithm is shown below:

<p align="center">
    <img src="docs/Fig4_algorithm_flowchart.png" alt="Algorithm flowchart" width="70%"/>
</p>

## Deep Learning
The repository includes four deep learning methods for binary segmentation:
- **PointNet**
- **PointNet++**
- **MinkUNet34C**
- **PointTransformerV3**

These models have been trained and evaluated with different feature configurations and optimisers. Some of the results obtained are shown below:

## Results

| Model               | Features   | Optimiser   | Scheduler   | F1 Score | mIoU   | TP        | FP      | TN        | FN      |
|---------------------|------------|-------------|-------------|----------|--------|-----------|---------|-----------|---------|
| PointTransformerV3  | XYZNXNYNZ  | AdamW       | OneCycleLR  | 98.79%   | 97.60% | 22,327,354 | 11,482  | 16,693,999 | 536,931 |
| PointTransformerV3  | XYZC       | AdamW       | OneCycleLR  | 98.46%   | 96.96% | 22,179,888 | 10,555  | 16,694,926 | 684,397 |
| PointTransformerV3  | XYZ        | AdamW       | OneCycleLR  | 98.72%   | 97.48% | 22,300,058 | 13,132  | 16,692,349 | 564,227 |
| PointTransformerV3  | NXNYNZ     | AdamW       | OneCycleLR  | 98.36%   | 96.78% | 22,138,548 | 10,702  | 16,694,779 | 725,737 |
| PointTransformerV3  | C          | AdamW       | OneCycleLR  | 99.35%   | 98.71% | 22,580,697 | 10,425  | 16,695,056 | 283,588 |
| PointNet++          | XYZNXNYNZ  | Adam        | Plateau     | 92.07%   | 85.31% | 20,303,511 | 936,668 | 15,768,813 | 2,560,774 |
| PointNet++          | XYZC       | Adam        | Plateau     | 97.09%   | 94.35% | 22,156,480 | 619,641 | 16,085,840 | 707,805 |
| PointNet++          | XYZ        | Adam        | Plateau     | 77.22%   | 62.89% | 15,880,131 | 2,387,160 | 14,318,321 | 6,984,154 |
| PointNet++          | NXNYNZ     | Adam        | Plateau     | 93.16%   | 87.20% | 20,998,161 | 1,215,742 | 15,489,739 | 1,866,124 |
| PointNet++          | C          | Adam        | Plateau     | 97.12%   | 94.41% | 22,026,361 | 466,385 | 16,239,096 | 837,924 |
| PointNet            | XYZNXNYNZ  | Adam        | Plateau     | 94.49%   | 89.55% | 10,056,968 | 231,333 | 7,949,302  | 942,397 |
| PointNet            | XYZC       | Adam        | Plateau     | 95.41%   | 91.22% | 10,154,349 | 132,728 | 8,047,907  | 845,016 |
| PointNet            | XYZ        | Adam        | Plateau     | 79.78%   | 66.36% | 7,520,098  | 333,594 | 7,847,041  | 3,479,267 |
| PointNet            | NXNYNZ     | Adam        | Plateau     | 62.51%   | 45.46% | 6,830,482  | 4,024,601 | 4,156,034  | 4,168,883 |
| PointNet            | C          | Adam        | Plateau     | 84.60%   | 73.31% | 9,087,651  | 1,396,327 | 6,784,308  | 1,911,714 |
| MinkUNet34C         | XYZNXNYNZ  | Adam        | Plateau     | 89.84%   | 81.56% | 19,882,413 | 1,513,231 | 15,192,250 | 2,981,872 |
| MinkUNet34C         | XYZC       | Adam        | Plateau     | 94.70%   | 89.93% | 21,130,736 | 633,579  | 16,071,902 | 1,733,549 |
| MinkUNet34C         | XYZ        | Adam        | Plateau     | 88.89%   | 80.01% | 19,426,171 | 1,415,812 | 15,289,669 | 3,438,114 |
| MinkUNet34C         | NXNYNZ     | Adam        | Plateau     | 83.58%   | 71.80% | 17,448,112 | 1,438,069 | 15,267,412 | 5,416,173 |
| MinkUNet34C         | C          | Adam        | Plateau     | 96.03%   | 92.36% | 21,978,497 | 931,985  | 15,773,496 | 885,788 |

## Credits

Developed by the ARVC group at Miguel Hernández University of Elche. For further information, please consult the included documentation or contact the authors.

## Citation

If you use this repository or any part of its content in your work, please cite the following article or use the reference below:

```bibtex
@article{Soler2025,
    AUTHOR = {Francisco J. Soler Mora, Adrián Peidró Vidal, Marc Fabregat-Jaén, Luis Payá Castelló, Óscar Reinoso García},
    TITLE = {Methods for the Segmentation of Reticular Structures Using 3D LiDAR Data: A Comparative Evaluation},
    JOURNAL = {Computer Modeling in Engineering \& Sciences},
    VOLUME = {143},
    YEAR = {2025},
    NUMBER = {3},
    PAGES = {3167--3195},
    URL = {http://www.techscience.com/CMES/v143n3/62813},
    ISSN = {1526-1506},
    DOI = {10.32604/cmes.2025.064510}
}
```

