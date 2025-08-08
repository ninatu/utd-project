#!/bin/bash

# === Select the dataset you want to process by uncommenting the corresponding block ===
dataset='msrvtt'
split='test'
add_keys='--retrieval'

#dataset='lsmdc'
#split='test'
#add_keys='--retrieval'
#
#dataset='didemo'
#split='test'
#add_keys='--retrieval'
#
#dataset='S-MiT'
#split='test'
#add_keys='--retrieval'
#
#dataset='youcook'
#split='val'
#add_keys='--retrieval'
#
#dataset='activity_net'
#split='val'
#add_keys='--retrieval'
#
#dataset='ucf'
#split='testlist01'
#add_keys=''
#
#dataset='ssv2'
#split='val'
#add_keys=''
#
#dataset='kinetics_400'
#split='val'
#add_keys=''
#
#dataset='kinetics_600'
#split='val'
#add_keys=''
#
#dataset='kinetics_700'
#split='val'
#add_keys=''
#
#dataset='MiT'
#split='val'
#add_keys=''


# Compute 'objects', 'activities', 'verbs' representation bias ('seq_of_f' temporal setup), as in Table 1:
temporal_aggregation_type='seq_of_f'
for concept in 'objects' 'activities' 'verbs'; do
  python utd/utd/compute_common_sense_bias.py \
        ${add_keys} \
        --dataset ${dataset} --split ${split} \
        --temporal_aggregation_type ${temporal_aggregation_type} \
        --concept ${concept} \
        --text_descriptions output/UTD_descriptions/${dataset}_${split}_${concept}.pickle \
        --output_path  output/bias/${dataset}_${split}_${concept}_${temporal_aggregation_type}.csv
done

# Compute objects+composition+activities bias in ('seq_of_f' temporal setup), as in Table 1:
concept='objects+composition+activities_15_words'
temporal_aggregation_type='seq_of_f'
python utd/utd/compute_common_sense_bias.py \
      ${add_keys} \
      --dataset ${dataset} --split ${split} \
      --temporal_aggregation_type ${temporal_aggregation_type} \
      --concept ${concept} \
      --text_descriptions output/UTD_descriptions/${dataset}_${split}_${concept}.pickle \
      --output_path  output/bias/${dataset}_${split}_${concept}_${temporal_aggregation_type}.csv


# ---------------------------------------------------------------------------------------------
# Compute representation bias for all (concept x temporal_aggregation_type) combinations  (Table A.3 of supplement)
# ---------------------------------------------------------------------------------------------


#for concept in 'objects' 'activities' 'verbs'; do
#  for temporal_aggregation_type in 'middle_f' 'max_score_f' 'avg_over_f' 'seq_of_f'; do
#    python utd/utd/compute_common_sense_bias.py \
#      ${add_keys} \
#      --dataset ${dataset} --split ${split} \
#      --temporal_aggregation_type ${temporal_aggregation_type} \
#      --concept ${concept} \
#      --text_descriptions output/UTD_descriptions/${dataset}_${split}_${concept}.pickle \
#      --output_path  output/bias/${dataset}_${split}_${concept}_${temporal_aggregation_type}.csv
#  done
#done
#
#concept='objects+composition+activities'
#for temporal_aggregation_type in 'middle_f' 'max_score_f' 'avg_over_f'; do
#    python utd/utd/compute_common_sense_bias.py \
#      ${add_keys} \
#      --dataset ${dataset} --split ${split} \
#      --temporal_aggregation_type ${temporal_aggregation_type} \
#      --concept ${concept} \
#      --text_descriptions output/UTD_descriptions/${dataset}_${split}_${concept}.pickle \
#      --output_path  output/bias/${dataset}_${split}_${concept}_${temporal_aggregation_type}.csv
#  done
#
#concept='objects+composition+activities_15_words'
#temporal_aggregation_type='seq_of_f'
#python utd/utd/compute_common_sense_bias.py \
#      ${add_keys} \
#      --dataset ${dataset} --split ${split} \
#      --temporal_aggregation_type ${temporal_aggregation_type} \
#      --concept ${concept} \
#      --text_descriptions output/UTD_descriptions/${dataset}_${split}_${concept}.pickle \
#      --output_path  output/bias/${dataset}_${split}_${concept}_${temporal_aggregation_type}.csv
