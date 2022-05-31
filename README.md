# SGPA: Structure-Guided Prior Adaptation for Category-Level 6D Object Pose Estimation
This is the PyTorch implemention of ICCV'21 paper **[SGPA: Structure-Guided Prior Adaptation for Category-Level 6D Object Pose Estimation](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_SGPA_Structure-Guided_Prior_Adaptation_for_Category-Level_6D_Object_Pose_Estimation_ICCV_2021_paper.pdf)** by Kai Chen and [Qi Dou](http://www.cse.cuhk.edu.hk/~qdou/).

<p align="center">
<img src="images/teaser.png" alt="intro" width="100%"/>
</p>

## Abstract
> Category-level 6D object pose estimation aims to predict the position and orientation for unseen objects, which plays a pillar role in many scenarios such as robotics and augmented reality. The significant intra-class variation is the bottleneck challenge in this task yet remains unsolved so far. In this paper, we take advantage of category prior to overcome this problem by innovating a structure-guided prior adaptation scheme to accurately estimate 6D pose for individual objects. Different from existing prior based methods, given one object and its corresponding category prior, we propose to leverage their structure similarity to dynamically adapt the prior to the observed object. The prior adaptation intrinsically associates the adopted prior with different objects, from which we can accurately reconstruct the 3D canonical model of the specific object for pose estimation. To further enhance the structure characteristic of objects, we extract low-rank structure points from the dense object point cloud, therefore more efficiently incorporating sparse structural information during prior adaptation. Extensive experiments on CAMERA25 and REAL275 benchmarks demonstrate significant performance improvement.

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
Download camera_train, camera_val, real_train, real_test, ground-truth annotations and mesh models provided by NOCS.

Then, organize and preprocess these files following [SPD](https://github.com/mentian/object-deformnet). For a quick evaluation, we provide the processed testing data for REAL275. You can download it [here](https://www.dropbox.com/s/7ylvilbwznme9cl/dataset.zip?dl=0) and organize the testing data as follows:
```
6D-CLGrasp
├── data
│   └── Real
│       ├──test
│       └──test_list.txt
└── results
    └── mrcnn_results
        └──real_test
```
## Evaluation
Please download our trained model [here](https://drive.google.com/file/d/1drBp3naBrNdazah1zTdgydjchE-LysdI/view?usp=sharing) and put it in the 'SGPA/model' directory. Then, you can have a quick evaluation on the REAL275 dataset using the following command.
```bash
bash eval.sh
```

## Train
In order to train the model, remember to download the complete dataset, organize and preprocess the dataset properly at first.

train.py is the main file for training. You can simply start training using the following command.
```bash
bash train.sh
```

## Citation
If you find the code useful, please cite our paper.
```latex
@inproceedings{chen2021sgpa,
  title={Sgpa: Structure-guided prior adaptation for category-level 6d object pose estimation},
  author={Chen, Kai and Dou, Qi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={2773--2782},
  year={2021}
}
```
Any questions, please feel free to contact Jian Liu (jianliu@hnu.edu.cn).

## Acknowledgment
Our code is developed based on the following repositories. We thank the authors for releasing the codes.
- [SGPA](https://github.com/leo94-hk/SGPA)
- [MHFormer](https://github.com/Vegetebird/MHFormer)
- [Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch)
