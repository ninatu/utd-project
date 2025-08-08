import os.path

import numpy as np
import pandas as pd
from argparse import ArgumentParser
import tqdm
import itertools
from ast import literal_eval

from utd.utd.utils.metrics import compute_retrieval_metrics, print_computed_retrieval_metrics
from utd.utd.utils.compute_bias_utils import get_description_instruct_prompt, get_label_instruct_prompt, \
    construct_instruction, aggregate_temporally, get_text_embeddings, normalize
from utd.utd.utils.utils import load_descriptions, pooling_embeddings


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Salesforce/SFR-Embedding-Mistral',
                        choices=['clip', 'longclip', 'intfloat/e5-mistral-7b-instruct',
                                 'Salesforce/SFR-Embedding-Mistral'],
                        help='The name of the model to use. Choices: "clip", "longclip", "intfloat/e5-mistral-7b-instruct", "Salesforce/SFR-Embedding-Mistral".')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the pre-trained LongCLIP model. This argument is only required if "longclip" is selected for --model_name.')

    parser.add_argument('--retrieval_dataset', action='store_true')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=["activity_net", "didemo", "kinetics_400", "kinetics_600", "kinetics_700",
                                 "lsmdc", "MiT", "msrvtt", "S-MiT", "ssv2", "ucf", "youcook"],
                        help='The dataset, choose from: "activity_net", '
                             '"didemo", "kinetics_400", "kinetics_600", "kinetics_700", "lsmdc", "MiT", "msrvtt", '
                             '"S-MiT", "ssv2", "ucf", "youcook".')
    parser.add_argument('--split', type=str, required=True,
                        help='The dataset split to use. Options: "test" or "val". '
                             'For ucf, use "testlist01" instead of "test".')

    parser.add_argument('--text_descriptions', help="Path to the input file containing textual descriptions.")
    parser.add_argument('--output_path', help='Path to save the output CSV file containing results for each video.')

    parser.add_argument('--concept', type=str, choices=['objects', 'activities', 'verbs',
                                 'objects+composition+activities','objects+composition+activities_15_words'])
    parser.add_argument('--temporal_aggregation_type', type=str, choices=['middle_f', 'max_score_f', 'avg_over_f', 'seq_of_f'])

    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    dataset = args.dataset
    split = args.split
    model_name = args.model_name
    concept = args.concept
    temporal_aggregation_type = args.temporal_aggregation_type
    output_path = args.output_path

    if model_name in ['clip', 'longclip']:
        if model_name == 'clip':
            import clip
            path = "ViT-L/14"
        else:
            from model import longclip as clip
            path = args.model_path

        feat_dim = 768
        model, preprocess = clip.load(path)
        model.cuda().eval()
        tokenize = clip.tokenize
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        tokenize = None
        feat_dim = 4096

    print("\n\nDataset:", dataset)
    print("Model name: ", model_name)
    print("Concept: ", concept)
    print("Temporal aggregation: ", temporal_aggregation_type)
    print("Output path:", output_path)

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

    concept_descriptions = {video_id: data[concept]
                            for video_id, data in concept_descriptions.items()}

    print("------------------------------------------------------------------")
    print(f"Example of concepts per video (video_id = {video_ids[0]}):")
    print(concept_descriptions[video_ids[0]])

    # ------------------------------- Compute embeddings for videos -------------------------------

    concept_descriptions = [concept_descriptions[video_id] for video_id in video_ids]
    descriptions = [aggregate_temporally(x, temporal_aggregation_type) for x in concept_descriptions]
    if model_name not in ['clip', 'longclip']:
        instruct = get_description_instruct_prompt(concept, temporal_aggregation_type, is_retrieval=args.retrieval_dataset)
        descriptions = [[construct_instruction(instruct, x) for x in concepts_per_video] for concepts_per_video in descriptions]

    print("------------------------------------------------------------------")
    print(f"Example of concepts per video after preprocessing (video_id = {video_ids[0]}):")
    for x in descriptions[0]:
        print(x)
        print('--------')
    print("------------------------------------------------------------------")

    # based on temporal agrregation, one video might have one or more descriptions
    n_per_video = [len(x) for x in descriptions]
    # merge all descriptions into one list for future embedding extractions
    all_descriptions = list(itertools.chain.from_iterable(descriptions))

    all_descriptions_embeddings = get_text_embeddings(model, model_name, tokenize, feat_dim, all_descriptions,
                                                      truncate=True,
                                                      batch_size=args.batch_size)
    # if max_score, we will pool later
    if temporal_aggregation_type != 'max_score_f':
        descriptions_embeddings = pooling_embeddings(all_descriptions_embeddings, n_per_video, pooling='mean')
    else:
        descriptions_embeddings = all_descriptions_embeddings

    # ------------------------------- Compute embeddings for gt caption/labels -------------------------------

    gt_captions_texts = gt_texts
    if model_name not in ['clip', 'longclip']:
        instruct = get_label_instruct_prompt(temporal_aggregation_type, is_retrieval=args.retrieval_dataset)
        gt_captions_texts = [[construct_instruction(instruct, x) for x in captions_per_video] for captions_per_video in gt_captions_texts]

    print("------------------------------------------------------------------")
    print(f"Example of gt captions  (video_id = {video_ids[0]}): {gt_captions_texts[0][:5]}")
    print("------------------------------------------------------------------")

    # one video might have several captions
    captions_per_video = [len(x) for x in gt_captions_texts]
    # merge all descriptions into one list for future embedding extractions
    all_gt_captions_texts = list(itertools.chain.from_iterable(gt_captions_texts))

    all_gt_captions_embeddings = get_text_embeddings(model, model_name, tokenize, feat_dim, all_gt_captions_texts,
                                                     truncate=True,
                                                     batch_size=args.batch_size)
    gt_captions_embeddings = pooling_embeddings(all_gt_captions_embeddings, captions_per_video)


    # ------------------------------- Compute similarities: (descriptions x labels/captions) -------------------------
    scores = np.matmul(normalize(descriptions_embeddings), normalize(gt_captions_embeddings).T)

    if temporal_aggregation_type == 'max_score_f':
        scores = pooling_embeddings(scores, n_per_video, pooling='max')

    # ------------------------------- Compute metrics, save cvs with predictions per video  -------------------------

    if args.retrieval_dataset:
        for retrieval_name, retrieval_scores in (("_tv", scores.T), ("_vt", scores)):
            metrics = compute_retrieval_metrics(retrieval_scores)

            print(dataset, concept, temporal_aggregation_type, retrieval_name,
                  f"{metrics['R1']:.3f} {metrics['R5']:.3f} {metrics['R10']:.3f}", '---------------------')
            print_computed_retrieval_metrics(metrics)

            df = []
            for correct_i, (correct_video, score, (_, item_info)) in tqdm.tqdm(enumerate(zip(concept_descriptions, retrieval_scores,
                                                                                          data_info.iterrows()))):
                video_id = item_info.video_id
                gt_caption = item_info.captions[0]

                sx = np.sort(-score)
                diff = sx - (-score[correct_i])
                predict_rank = np.mean(np.argwhere(diff == 0)) + 1

                ranks = (-score).argsort()
                predict_text = gt_texts[ranks[0]][0]
                predict_video = descriptions[ranks[0]]

                df.append((video_id, gt_caption, correct_video, predict_rank, predict_video, predict_text))

            df = pd.DataFrame(df, columns=['video_id', 'caption', 'video_description',
                                           'predict_rank', 'descriptions_of_top1_predict',
                                           'caption_of_top1_predict'])

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_path = os.path.splitext(output_path)[0] + retrieval_name + os.path.splitext(output_path)[1]
            df.to_csv(save_path, index=False)

    else:
        classes_info_dict = {i: (item['class_name'], item['class_text']) for i, (_, item) in
                             enumerate(classes_info.iterrows())}

        df = []
        for input_text, score, (_, item_info) in tqdm.tqdm(zip(descriptions, scores, data_info.iterrows())):
            correct = item_info.class_name
            correct_text = item_info.class_text
            video_id = item_info.video_id

            predict = [classes_info_dict[i][0] for i in (-score).argsort()]
            predict_text = [classes_info_dict[i][1] for i in (-score).argsort()]

            df.append((video_id, correct, correct_text, input_text, predict, predict_text ))

        df = pd.DataFrame(df, columns=['video_id', 'gt_label', 'gt_label_text', 'video_description',
                                       'predict_labels', 'predict_label_texts'])
        correct = df.apply(axis=1, func=lambda x: x['gt_label'] in x['predict_labels'][:1])
        df['Correct'] = None
        df.loc[correct, 'Correct'] = 'Correct'

        accuracy = (df.apply(axis=1, func=lambda x: x['gt_label'] in x['predict_labels'][:1])).mean()
        accuracy_top5 = (df.apply(axis=1, func=lambda x: x['gt_label'] in x['predict_labels'][:5])).mean()
        accuracy_top10 = (df.apply(axis=1, func=lambda x: x['gt_label'] in x['predict_labels'][:10])).mean()

        print(dataset, concept, temporal_aggregation_type,
              'acc_top1: {:.3f}, acc_top5: {:.3f}, acc_top10: {:.3f}'.format(accuracy, accuracy_top5, accuracy_top10))
        print(dataset, concept, temporal_aggregation_type
              , '{:.3f} {:.3f} {:.3f}'.format(accuracy, accuracy_top5, accuracy_top10))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
