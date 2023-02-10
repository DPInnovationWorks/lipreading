#!/bin/bash
#SBATCH --job-name=lrs2_face_recognition
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --comment=WHMFC_news

# activate conda env
source /group_homes/public_cluster/home/share/LipReadingGroup/enviroment/.bashrc
env
nvcc --version
nvidia-smi
which conda
which python
conda activate cuda11.7-torch1.13
cd ~/share/LipReadingGroup/lipreading
which python
python -u preprocess/lrs2/lrs2_face_recognition.py