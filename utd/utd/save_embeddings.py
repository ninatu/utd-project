import os

import numpy as np
import pandas as pd
from argparse import ArgumentParser
from ast import literal_eval
from sentence_transformers import SentenceTransformer

from utd.utd.utils.compute_bias_utils import get_description_instruct_prompt, get_label_instruct_prompt, \
    construct_instruction, aggregate_temporally, get_pooled_text_embeddings
from utd.utd.utils.utils import load_descriptions


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Salesforce/SFR-Embedding-Mistral',
                        choices=['intfloat/e5-mistral-7b-instruct',
                                 'Salesforce/SFR-Embedding-Mistral'],
                        help='The name of the model to use. Choices: "intfloat/e5-mistral-7b-instruct", "Salesforce/SFR-Embedding-Mistral".')

    parser.add_argument('--retrieval_dataset', action='store_true')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=["kinetics_400", "kinetics_600", "kinetics_700", "MiT", "ssv2", "ucf"],
                        help='The dataset, choose from: "kinetics_400", "kinetics_600", "kinetics_700", "MiT", "ssv2", "ucf".')
    parser.add_argument('--split', type=str, required=True, help='The dataset split to use.')
    parser.add_argument('--text_descriptions', help="Path to the input file containing textual descriptions.")

    parser.add_argument('--output_desc_emb_path')
    parser.add_argument('--dont_compute_labels_emb', action='store_true')
    parser.add_argument('--output_labels_emb_path', default=None)

    parser.add_argument('--concept', type=str, choices=['objects', 'activities', 'verbs',
                                 'objects+composition+activities','objects+composition+activities_15_words'])
    parser.add_argument('--temporal_aggregation_type', type=str, default='seq_of_f',
                        choices=['seq_of_f'])
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()

    dataset = args.dataset
    split = args.split
    model_name = args.model_name
    concept = args.concept
    temporal_aggregation_type = args.temporal_aggregation_type

    if not args.dont_compute_labels_emb:
        assert args.dont_compute_labels_emb is not None

    model = SentenceTransformer(model_name)
    tokenize = None
    feat_dim = 4096

    print("\n\nDataset:", dataset)
    print("Model name: ", model_name)
    print("Concept: ", concept)
    print("Temporal aggregation: ", temporal_aggregation_type)
    print("Output path for description embeddings:", args.output_desc_emb_path)
    print("Output path for labels/captions embeddings:", args.output_labels_emb_path)

    data_info = pd.read_csv(f'metadata/{dataset}/{split}_info.csv')
    data_info.index = data_info['video_id']
    video_ids = data_info['video_id'].tolist()
    if args.retrieval_dataset:
        if 'captions' in data_info:
            data_info.captions = data_info.captions.apply(literal_eval)
        else:
            data_info['captions'] = data_info.caption.apply(lambda x: [x])

        gt_texts = data_info['captions'].tolist()
    else:
        classes_info = pd.read_csv(f'metadata/{dataset}/classes.csv')
        classes_info.index = classes_info['class_name']

        gt_texts = classes_info['class_text'].tolist()
        gt_texts = [[x] for x in gt_texts]

    concept_descriptions = load_descriptions(args.text_descriptions)

    concept_descriptions = {video_id: data[concept] for video_id, data in concept_descriptions.items()}

    # ------------------------------- Compute embeddings for gt caption/labels -------------------------------

    if not args.dont_compute_labels_emb:
        gt_captions_texts = gt_texts
        instruct = get_label_instruct_prompt(temporal_aggregation_type, is_retrieval=args.retrieval_dataset)
        gt_captions_texts = [[construct_instruction(instruct, x) for x in captions_per_video] for captions_per_video in gt_captions_texts]

        gt_captions_embeddings = get_pooled_text_embeddings(model, model_name, tokenize, feat_dim, gt_captions_texts,
                                                             truncate=True,
                                                             batch_size=args.batch_size)
        os.makedirs(os.path.dirname(args.output_labels_emb_path), exist_ok=True)
        np.savez(args.output_labels_emb_path, gt_captions_embeddings)

    # ------------------------------- Compute embeddings for videos -------------------------------

    assert temporal_aggregation_type != 'max_score_f'
    concept_descriptions = [concept_descriptions[video_id] for video_id in video_ids]
    descriptions = [aggregate_temporally(x, temporal_aggregation_type) for x in concept_descriptions]

    instruct = get_description_instruct_prompt(concept, temporal_aggregation_type, is_retrieval=args.retrieval_dataset)
    descriptions = [[construct_instruction(instruct, x) for x in concepts_per_video] for concepts_per_video in descriptions]

    descriptions_embeddings = get_pooled_text_embeddings(model, model_name, tokenize, feat_dim, descriptions, truncate=True,
                                                  batch_size=args.batch_size)
    os.makedirs(os.path.dirname(args.output_desc_emb_path), exist_ok=True)
    np.savez(args.output_desc_emb_path, descriptions_embeddings)



