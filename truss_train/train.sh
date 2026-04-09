#!/bin/bash
set -eux

# torch is pre-installed in the pytorch/pytorch base image — skip it here
pip install "torchvision>=0.16.0" "Pillow>=9.3.0" "tqdm>=4.65.0"

python train.py
