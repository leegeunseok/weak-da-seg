#!/bin/bash

# model
model="deeplab"

# dataset
source="yang_s"
target="crack500_905_t"
# gaps_s, gaps_t, road_s, road_t, deepcrack_s, deepcrack_t, deepnon_s, deepnon_t
# crack500_s, crack500_t, crack500non_s, crack500non_t, crack500_305_s, crack500_305_t, crack500_605_s, crack500_605_t, crack500_905_s, crack500_905_t, rissbilder_s, rissbilder_t
# generalcon_s, generalcon_t
# kaggle_s, kaggle_t
# yang_s, yang_t

if [ "$source" = "yang_s" ]
then
    source_path="/home/user/WindowsShare/05. Data/00. Benchmarks/26. Crack_Yang/01.convert2cityscapes"
fi
# /home/user/WindowsShare/05. Data/00. Benchmarks/22. KhanhhaCrack/04. convert2cityscapes, /home/user/WindowsShare/05. Data/00. Benchmarks/22. KhanhhaCrack/06. dilation
# /home/user/WindowsShare/05. Data/00. Benchmarks/27. crackseg9k/convert2cityscapes, /home/user/WindowsShare/05. Data/00. Benchmarks/27. crackseg9k/add_noncrack, /home/user/WindowsShare/05. Data/00. Benchmarks/27. crackseg9k/part/crack500_305
# /home/user/WindowsShare/05. Data/02. Training&Test/013. General Concrete Damage/01. Cityscapes/v_bh
# /home/user/WindowsShare/05. Data/00. Benchmarks/23. KaggleCrack/06. dilation
# /home/user/WindowsShare/05. Data/00. Benchmarks/26. Crack_Yang/01.convert2cityscapes

if [ "$target" = "crack500_905_t" ]
then
    target_path="/home/user/WindowsShare/05. Data/00. Benchmarks/27. crackseg9k/part/crack500_905"
fi

# 448,448
# 400,400
# 1024,1024
# 1008,756
# 306,306

if [ "$source" = "yang_s" ]
then
    source_size="306,306"
fi

if [ "$target" = "crack500_905_t" ]
then
    target_size="400,400"
fi

if [ "$source" = "yang_s" ]
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
lambda_weak2=0.2  # 0.01 for pseudo, 0.2 for oracle
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
