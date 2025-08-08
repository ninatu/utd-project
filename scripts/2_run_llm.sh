#!/bin/bash


# === Select the dataset you want to process by uncommenting the corresponding block ===
dataset='msrvtt'
splits=('test' 'train')

#dataset='lsmdc'
#splits=('test' 'train')

#dataset='didemo'
#splits=('test' 'train')

#dataset='S-MiT'
#splits=('test' 'train_subset')

#dataset='youcook'
#splits=('val' 'train')

#dataset='activity_net'
#splits=('val' 'train')

#dataset='ucf'
#splits=('testlist01' 'trainlist01')

#dataset='ssv2'
#splits=('val' 'train')

#dataset='kinetics_400'
#splits=('val' 'train')

#dataset='kinetics_600'
#splits=('val' 'train')

#dataset='kinetics_700'
#splits=('val' 'train')

#dataset='MiT'
#splits=('val' 'train_subset')


for split in "${splits[@]}"; do
  for concept in "objects" "activities"; do
    python utd/utd/llm_extract_concepts.py  \
      --concept ${concept} \
      --input_description_path output/UTD_descriptions/${dataset}_${split}_objects+composition+activities.pickle \
      --output_path output/UTD_descriptions/${dataset}_${split}_${concept}.pickle \
      --save_raw_llm_output_path output/UTD_descriptions/raw_${dataset}_${split}_${concept}.pickle
  done

  concept='verbs'
  python utd/utd/llm_extract_concepts.py  \
    --concept ${concept} \
    --input_description_path output/UTD_descriptions/raw_${dataset}_${split}_activities.pickle \
    --output_path output/UTD_descriptions/${dataset}_${split}_${concept}.pickle

  concept='objects+composition+activities_15_words'
  python utd/utd/llm_extract_concepts.py  \
    --concept ${concept} \
    --input_description_path output/UTD_descriptions/${dataset}_${split}_objects+composition+activities.pickle \
    --output_path output/UTD_descriptions/${dataset}_${split}_${concept}.pickle
done

