#!/bin/bash

# === Select the dataset you want to process by uncommenting the corresponding block ===

dataset='msrvtt'
dataset_root='data/msrvtt/'
splits=('test' 'train')

#dataset='lsmdc'
#dataset_root='data/lsmdc/'
#splits=('test' 'train')

#dataset='didemo'
#dataset_root='data/didemo/'
#splits=('test' 'train')

#dataset='S-MiT'
#dataset_root='data/S-MiT/'
#splits=('test' 'train_subset')

#dataset='youcook'
#dataset_root='data/youcook/'
#splits=('val' 'train')

#dataset='activity_net'
#dataset_root='data/youcook/'
#splits=('val' 'train')

#dataset='ucf'
#dataset_root='data/UCF101/'
#splits=('testlist01' 'trainlist01')

#dataset='ssv2'
#dataset_root='data/SomethingSomething_v2/'
#splits=('val' 'train')

#dataset='kinetics_400'
#dataset_root='data/kinetics_400/'
#splits=('val' 'train')

#dataset='kinetics_600'
#dataset_root='data/kinetics_600/'
#splits=('val' 'train')

#dataset='kinetics_700'
#dataset_root='data/kinetics_700/'
#splits=('val' 'train')

#dataset='MiT'
#dataset_root='data/MiT/'
#splits=('val' 'train_subset')


for split in "${splits[@]}"; do
  python utd/utd/vlm_descriptions_llava.py \
    --dataset ${dataset} --split ${split} --dataset_root ${dataset_root} --metadata_root metadata/${dataset} \
    --output_path output/UTD_descriptions/${dataset}_${split}_objects+composition+activities.pickle
done



