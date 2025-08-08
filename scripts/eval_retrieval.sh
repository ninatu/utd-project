#!/bin/bash

# =====================================
# 🔧 Select the model (uncomment one)
# =====================================

model_keys="--backbone_name umt-b-5M --backbone umt --model vit_b16 --backbone_checkpoint pretrained/umt/b16_5m.pth"
#model_keys="--backbone_name umt-b-17M --backbone umt --model vit_b16 --backbone_checkpoint pretrained/umt/b16_17m.pth"
#model_keys="--backbone_name umt-b-25M --backbone umt --model vit_b16 --backbone_checkpoint pretrained/umt/b16_25m.pth"
#model_keys="--backbone_name umt-l-5M --backbone umt --model vit_l14 --backbone_checkpoint pretrained/umt/l16_5m.pth"
#model_keys="--backbone_name umt-l-17M --backbone umt --model vit_l14 --backbone_checkpoint pretrained/umt/l16_17m.pth"
#model_keys="--backbone_name umt-l-25M --backbone umt --model vit_l14 --backbone_checkpoint pretrained/umt/l16_25m.pth"
#model_keys="--backbone_name internvid-B-10M-FLT --backbone internvid --model b --backbone_checkpoint pretrained/internvid/ViCLIP-B_InternVid-FLT-10M_fixed.pth --no_itm"
#model_keys="--backbone_name internvid-B-200M --backbone internvid --model b --backbone_checkpoint pretrained/internvid/ViCLIP-B_InternVid-200M_fixed.pth --no_itm"
#model_keys="--backbone_name internvid-L-200M --backbone internvid --model l --backbone_checkpoint pretrained/internvid/ViCLIP-L_InternVid-200M_fixed.pth --no_itm"
#model_keys="--backbone_name internvid-L-50M --backbone internvid --model l --backbone_checkpoint pretrained/internvid/ViCLIP-L_InternVid-50M_fixed.pth --no_itm"
#model_keys="--backbone_name internvid-L-10M --backbone internvid --model l --backbone_checkpoint pretrained/internvid/ViCLIP-L_InternVid-10M_fixed.pth --no_itm"
#model_keys="--backbone_name internvid-L-WebVid10M --backbone internvid --model l --backbone_checkpoint pretrained/internvid/ViCLIP-L_WebVid-10M_fixed.pth --no_itm"
#model_keys="--backbone_name internvid-L-10M-DIV --backbone internvid --model l --backbone_checkpoint pretrained/internvid/ViCLIP-L_InternVid-DIV-10M_fixed.pth --no_itm"
#model_keys="--backbone_name internvid-L-10M-FLT --backbone internvid --model l --backbone_checkpoint pretrained/internvid/ViClip-InternVid-10M-FLT_fixed.pth --no_itm"
#model_keys="--backbone_name videomamba-vm-5M --backbone videomamba --backbone_checkpoint pretrained/videomamba/videomamba_m16_5M_f8_res224.pth"
#model_keys="--backbone_name videomamba-vm-17M --backbone videomamba --backbone_checkpoint pretrained/videomamba/videomamba_m16_17M_f8_res224.pth"
#model_keys="--backbone_name videomamba-vm-25M --backbone videomamba --backbone_checkpoint pretrained/videomamba/videomamba_m16_25M_f8_res224.pth"


# =====================================
# 🔧 Select the dataset (uncomment one)
# =====================================

dataset_keys='--dataset msrvtt --split test --dataset_root data/msrvtt/ --dataset_metadata_root  metadata/msrvtt/ --debiased_splits_path UTD_splits/splits_msrvtt_test.json'
#dataset_keys='--dataset youcook --split val --dataset_root data/youcook/ --dataset_metadata_root  metadata/youcook/ --debiased_splits_path UTD_splits/splits_youcook_val.json'
#dataset_keys='--dataset didemo --split test --dataset_root data/didemo/ --dataset_metadata_root  metadata/didemo/ --debiased_splits_path UTD_splits/splits_didemo_test.json'
#dataset_keys='--dataset lsmdc --split test --dataset_root data/lsmdc/ --dataset_metadata_root  metadata/lsmdc/ --debiased_splits_path UTD_splits/splits_lsmdc_test.json'
#dataset_keys='--dataset activity_net --split val --dataset_root data/activity_net/ --dataset_metadata_root  metadata/activity_net/ --debiased_splits_path UTD_splits/splits_activity_net_val.json'
#dataset_keys='--dataset S-MiT --split test --dataset_root data/S-MiT/ --dataset_metadata_root  metadata/S-MiT/ --debiased_splits_path UTD_splits/splits_S-MiT_test.json'

export PYTHONPATH="./third_party/:$PYTHONPATH"
python utd/videoglue/eval_retrieval.py ${model_keys} $dataset_keys \
      --encoder_batch_size 32 --fusion_batch_size 32 \
      --output_root 'output/retrieval'
