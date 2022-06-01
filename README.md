# Real-Time Multi-Object Robotic Grasping System by Shape Transformer-Guided Category-Level 6D Pose Estimation
This is the PyTorch implemention of our paper **[Real-Time Multi-Object Robotic Grasping System by Shape Transformer-Guided Category-Level 6D Pose Estimation]()** published in <b>*IEEE Transactions on Industrial Informatics*</b> by J. Liu, W. Sun, C. Liu, X. Zhang, and Q. Fu.

<p align="center">
<img src="images/teaser.png" alt="intro" width="100%"/>
</p>

## Abstract
> Category-level 6D object pose estimation aims to predict the position and orientation for unseen objects, which plays a pillar role in many scenarios such as robotics and augmented reality. The significant intra-class variation is the bottleneck challenge in this task yet remains unsolved so far. In this paper, we take advantage of category prior to overcome this problem by innovating a structure-guided prior adaptation scheme to accurately estimate 6D pose for individual objects. Different from existing prior based methods, given one object and its corresponding category prior, we propose to leverage their structure similarity to dynamically adapt the prior to the observed object. The prior adaptation intrinsically associates the adopted prior with different objects, from which we can accurately reconstruct the 3D canonical model of the specific object for pose estimation. To further enhance the structure characteristic of objects, we extract low-rank structure points from the dense object point cloud, therefore more efficiently incorporating sparse structural information during prior adaptation. Extensive experiments on CAMERA25 and REAL275 benchmarks demonstrate significant performance improvement.

## Citation
If you find the code useful, please cite our paper.
```latex
@article{TII2022,
  author={Liu, Jian and Sun, Wei and Liu, Chongpei and Zhang, Xing and Fu, Qiang},
  journal={IEEE Transactions on Industrial Informatics},
  title={Real-Time Multi-Object Robotic Grasping System by Shape Transformer-Guided Category-Level 6D Pose Estimation},
  year={2022},
  publisher={IEEE}
}

```
Any questions, please feel free to contact Jian Liu (jianliu@hnu.edu.cn).

## Installation
Our code has been tested with
- Ubuntu 20.04
- Python 3.8
- CUDA 11.0
- PyTorch 1.8.0

We recommend using conda to setup the environment.

If you have already installed conda, please use the following commands.

```bash
conda create -n CLGrasp python=3.8
conda activate CLGrasp
pip install ...
```
**Build PointNet++**

```bash
cd 6D-CLGrasp/pointnet2/pointnet2
python setup.py install
```
**Build nn_distance**

```bash
cd 6D-CLGrasp/lib/nn_distance
python setup.py install
```

## Dataset
Download [camera_train](http://download.cs.stanford.edu/orion/nocs/camera_train.zip), [camera_val](http://download.cs.stanford.edu/orion/nocs/camera_val25K.zip),
[real_train](http://download.cs.stanford.edu/orion/nocs/real_train.zip), [real_test](http://download.cs.stanford.edu/orion/nocs/real_test.zip),
[ground-truth annotations](http://download.cs.stanford.edu/orion/nocs/gts.zip),
and [mesh models](http://download.cs.stanford.edu/orion/nocs/obj_models.zip)
provided by [NOCS](https://github.com/hughw19/NOCS_CVPR2019).<br/>
Unzip and organize these files in 6D-CLGrasp/data as follows:
```
data
├── CAMERA
│   ├── train
│   └── val
├── Real
│   ├── train
│   └── test
├── gts
│   ├── val
│   └── real_test
└── obj_models
    ├── train
    ├── val
    ├── real_train
    └── real_test
```
Run python scripts to prepare the datasets.
```
cd 6D-CLGrasp/preprocess
python shape_data.py
python pose_data.py
```

## Evaluation
You can download our pretrained models [here](https://drive.google.com/file/d/1drBp3naBrNdazah1zTdgydjchE-LysdI/view?usp=sharing) and put them in the '6D-CLGrasp/train_results/CAMERA' and the '6D-CLGrasp/train_results/REAL' directory, respectively. Then, you can have a quick evaluation on the CAMERA25 and REAL275 datasets using the following command.
```bash
bash eval.sh
```

## Train
In order to train the model, remember to download the complete dataset, organize and preprocess the dataset properly at first.
```
# optional - train the GSENet and to get the global shapes (the pretrained global shapes can be found in '6D-CLGrasp/assets1')
python train_ae.py
python mean_shape.py
```

train.py is the main file for training. You can simply start training using the following command.
```bash
bash train.sh
```

## Acknowledgment
Our code is developed based on the following repositories. We thank the authors for releasing the codes.
- [SGPA](https://github.com/leo94-hk/SGPA)
- [SPD](https://github.com/mentian/object-posenet)
- [MHFormer](https://github.com/Vegetebird/MHFormer)
- [Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch)

## Licence

This project is licensed under the terms of the MIT license.
