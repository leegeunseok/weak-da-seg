#!/bin/bash

# model
model="deeplab"

# dataset
source="cfd_s"
target="cfd_t"
# gaps_s, gaps_t, cfd_s, cfd_t, deepcrack_s, deepcrack_t, cracktree200_s, cracktree200_t
# crack500_s, crack500_t, rissbilder_s, rissbilder_t
# kaggle_s, kaggle_t
# yang_s, yang_t
# road_s, road_t
# concrete_s, concrete_t

if [ "$source" = "cfd_s" ]
then
    source_path="/home/user/WindowsShare/05. Data/00. Benchmarks/22. KhanhhaCrack/04. convert2cityscapes"
fi
# /home/user/WindowsShare/05. Data/00. Benchmarks/22. KhanhhaCrack/04. convert2cityscapes
# /home/user/WindowsShare/05. Data/00. Benchmarks/27. crackseg9k/convert2cityscapes
# /home/user/WindowsShare/05. Data/00. Benchmarks/23. KaggleCrack/03. Cityscapes Dataset
# /home/user/WindowsShare/05. Data/00. Benchmarks/26. Crack_Yang/01.convert2cityscapes
# /home/user/WindowsShare/05. Data/00. Benchmarks/33. DA/road
# /home/user/WindowsShare/05. Data/00. Benchmarks/33. DA/concrete

if [ "$target" = "cfd_t" ]
then
    target_path="/home/user/WindowsShare/05. Data/00. Benchmarks/22. KhanhhaCrack/04. convert2cityscapes"
fi

# 448,448
# 400,400
# 1008,756
# 306,306
# 448,448
# 448,448

if [ "$source" = "cfd_s" ]
then
    source_size="448,448"
fi

if [ "$target" = "cfd_t" ]
then
    target_size="448,448"
fi

if [ "$source" = "cfd_s" ]
then
    num_classes=2
fi

source_split="train"
target_split="train"
test_split="val"

# parameters
batch_size=1
num_steps=121000
num_steps_stop=121000

lambda_seg=0.0
lambda_adv1=0.0
lambda_adv2=0.001  # 0.001
lambda_weak2=0.01  # 0.01 for pseudo, 0.2 for oracle
lambda_weak_cwadv2=0.001  # 0.001

lr=2.5e-4
# lr=0.00001
lr_d=1e-4

save_step=10000
print_step=100
pweak_th=0.2

#
# training models
#
# - use [--val] to run testing during training for selecting better models (require a validation set in the target domain with ground truths)
# - If `--val-only` is added, the code runs the testing mode.

## training models when GTA is the source domain
# pretrain="models/"$source"_pretrained.pth"

## training models when other dataset is the source domain ##
pretrain="models/MS_DeepLab_resnet_pretrained_COCO_init.pth"
# pretrain=True

#
# testing models
#   NB: Do not forget to add the argument `--val-only` below
#
# pretrain="model/gta5-cityscapes-pseudoweak-cw.pth"
