#!/bin/bash

set -x
set -e


export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=1
#change this!!!!!! camera_val(CAMERA) or real_test(Real)
python evaluate.py --data real_test\
  --model train_results/REAL/model.pth\
  --num_structure_points 256\
  --result_dir results/REAL_at+globalshape
  
