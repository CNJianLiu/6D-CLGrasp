#!/bin/bash

set -x
set -e

#export TORCH_CUDA_ARCH_LIST="8.0"
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=3

# python train.py --dataset CAMERA+Real\
#   --result_dir train_results/real_at+globalshape/


python train.py --dataset CAMERA\
  --result_dir train_results/camera_at+globalshape/

