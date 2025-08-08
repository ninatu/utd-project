#!/bin/bash

# ================================
#  neptune.ai keys (optional)
# ================================

neptune_api_token=""
neptune_project=""

# ================================
#  Dataset Configuration: Uncomment ONE dataset block
# ================================
dataset_name='ucf'
prefix='data/UCF101/original/videos/'
data_path='metadata/videoglue/ucf/'
dataset_type='Kinetics_sparse'
debiased_splits_path='UTD_splits/splits_ucf_testlist01.json'
nb_classes=101

#dataset_name='ssv2'
#prefix='data/SomethingSomething_v2'
#data_path='metadata/videoglue/ssv2/'
#debiased_splits_path='UTD_splits/splits_ssv2_val.json'
#dataset_type='SSV2'
#nb_classes=174
#
#dataset_name='kinetics_400'
#prefix='data/kinetics_400'
#data_path='metadata/videoglue/kinetics_400/'
#debiased_splits_path='UTD_splits/splits_kinetics_400_val.json'
#dataset_type='Kinetics_sparse'
#nb_classes=400
#
#dataset_name='kinetics_600'
#prefix='data/kinetics_600'
#data_path='metadata/videoglue/kinetics_600/'
#debiased_splits_path='UTD_splits/splits_kinetics_600_val.json'
#dataset_type='Kinetics_sparse'
#nb_classes=600
#
#dataset_name='kinetics_700'
#prefix='data/kinetics_700'
#data_path='metadata/videoglue/kinetics_700/'
#debiased_splits_path='UTD_splits/splits_kinetics_700_val.json'
#dataset_type='Kinetics_sparse'
#nb_classes=700
#
#dataset_name='MiT'
#prefix='data/MiT'
#data_path='metadata/videoglue/MiT/'
#debiased_splits_path='UTD_splits/splits_MiT_val.json'
#dataset_type='Kinetics_sparse'
#nb_classes=305

# ================================
#  Model Configuration: Uncomment ONE backbone block
# ================================

backbone_name='videomae-B-K400'
backbone_parameters='--backbone videomae --model vit_base_patch16_224  --num_frames 16 --backbone_checkpoint pretrained/videomae/videomae_checkpoint_pretrain_kin400.pth'

#backbone_name='allinone-WV2M+CC'
#backbone_parameters='--backbone allinone --num_frames 8 --backbone_checkpoint pretrained/allinone/all-in-one-plus-base_fixed.ckpt'

#backbone_name='umt-L-K710-fnK710'
#backbone_parameters='--backbone umt --model vit_large_patch16_224 --num_frames 8 --backbone_checkpoint pretrained/umt/l16_ptk710_ftk710_f8_res224.pth'

#backbone_name='videomamba-vm-K400'
#backbone_parameters='--backbone videomamba --num_frames 8 --deepspeed_mixed_precision_type bf16 --backbone_checkpoint pretrained/videomamba/videomamba_m16_k400_mask_pt_f8_res224.pth'

#backbone_name='internvid-L-200M'
#backbone_parameters='--backbone internvid --model l  --num_frames 8 --backbone_checkpoint pretrained/internvid/ViCLIP-L_InternVid-200M_fixed.pth'

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

export PYTHONPATH="./third_party/:$PYTHONPATH"

srun python utd/videoglue/train_classifier.py \
        ${backbone_parameters} \
        --exp_name ${dataset_name}_${backbone_name}  \
        \
        --data_path ${data_path} \
        --prefix ${prefix} \
        --data_set ${dataset_type} \
        --debiased_splits_path ${debiased_splits_path} \
        --split ',' \
        --nb_classes ${nb_classes} \
        --log_dir 'output/videoglue/logs' \
        --output_dir 'output/videoglue/models' \
        \
        --ep 50 \
        --lr 0.001 \
        --weight_decay 0.05 \
        --batch_size 64 \
        --val_batch_size_mul 0.5 \
        --num_sample 1 \
        --input_size 224 \
        --short_side_size 224 \
        --num_workers 16 \
        \
        --aspect_ratio 0.5 2.0 \
        --area_ratio 0.3 1.0 \
        --aa rand-m9-n2-mstd0.5 \
        --reprob 0 \
        --mixup 0.8 \
        --cutmix 1.0 \
        \
        --warmup_epochs 5 \
        --opt adamw \
        --opt_betas 0.9 0.999 \
        \
        --no_test \
        --dist_eval \
        --enable_deepspeed \
        --layer_decay 1.0 \
        \
        --enable_neptune \
        --neptune_api_token ${neptune_api_token} \
        --neptune_project ${neptune_project}
        # disable the last block if you don't want to turn on neptune logging