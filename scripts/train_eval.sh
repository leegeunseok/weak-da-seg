conda activate da_seg_py38
cd weak-da-seg-new
cd weak-da-seg

SCRIPT_DIR="/home/user/weak-da-seg-new/weak-da-seg/scripts"
source $SCRIPT_DIR/common_crack_to_crack.sh

snapshot_dir="snapshot/"$source"-"$target"-point"
result_dir="result/"$source"-"$target"-point"

CUDA_VISIBLE_DEVICES=1 python train.py --model $model --dataset-source $source --dataset-target $target --data-path-source "/home/user/WindowsShare/05. Data/00. Benchmarks/26. Crack_Yang/01.convert2cityscapes" --data-path-target "/home/user/WindowsShare/05. Data/00. Benchmarks/27. crackseg9k/part/crack500_905" --input-size-source $source_size --input-size-target $target_size --num-classes $num_classes --source-split $source_split --target-split $target_split --test-split $test_split --batch-size $batch_size --num-steps $num_steps --num-steps-stop $num_steps_stop --lambda-seg $lambda_seg --lambda-adv-target1 $lambda_adv1 --lambda-adv-target2 $lambda_adv2 --lambda-weak-cwadv2 $lambda_weak_cwadv2 --lambda-weak-target2 $lambda_weak2 --learning-rate $lr --learning-rate-D $lr_d --restore-from $pretrain --pweak-th $pweak_th --snapshot-dir $snapshot_dir --result-dir $result_dir --save-pred-every $save_step --print-loss-every $print_step --use-weak --use-pseudo --use-weak-cw --use-pointloss --use-pixeladapt --val

python eval.py --model deeplab --dataset-source crack500_s --dataset-target deepcrack_t --data-path-source "/home/user/WindowsShare/05. Data/00. Benchmarks/27. crackseg9k/convert2cityscapes" --data-path-target "/home/user/WindowsShare/05. Data/00. Benchmarks/22. KhanhhaCrack/06. dilation" --input-size-source "400,400" --input-size-target "448,448" --num-classes 2 --test-split val --restore-from "" --result-dir ""