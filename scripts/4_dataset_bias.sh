#!/bin/bash


dataset="ucf"
temporal_aggregation_type="seq_of_f"
concept="objects"
test_split="testlist01"
train_split="trainlist01"

for split in ${test_split} ${train_split}; do
  python utd/utd/save_embeddings.py \
    --dataset ${dataset} --split ${split} \
    --temporal_aggregation_type ${temporal_aggregation_type} \
    --concept ${concept} \
    --batch_size 8 \
    --text_descriptions output/UTD_descriptions/${dataset}_${split}_${concept}.pickle \
    --output_desc_emb_path  output/embeddings/${dataset}_${split}_${concept}_${temporal_aggregation_type}.npz \
    --output_labels_emb_path  output/embeddings/${dataset}_${split}_labels.npz
done

python utd/utd/compute_dataset_bias.py \
  --dataset ${dataset} --train_split ${train_split} --test_split ${test_split} \
  --train_embeddings_path output/embeddings/${dataset}_${train_split}_${concept}_${temporal_aggregation_type}.npz \
  --test_embeddings_path output/embeddings/${dataset}_${test_split}_${concept}_${temporal_aggregation_type}.npz \
  --output_path  output/bias/dataset_bias_${dataset}_${split}_${concept}_${temporal_aggregation_type}.npz \

