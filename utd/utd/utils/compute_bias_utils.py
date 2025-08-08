import itertools

import numpy as np
import torch
import tqdm

from utd.utd.utils.utils import pooling_embeddings


def get_label_instruct_prompt(temporal_aggregation_type, is_retrieval):
    assert temporal_aggregation_type in ['middle_f', 'max_score_f', 'avg_over_f', 'seq_of_f']
    if is_retrieval:
        if temporal_aggregation_type == 'seq_of_f':
            class_instruct = 'Given a short video description, retrieve another description of this video.'
        else:
            class_instruct = 'Given a short video description, retrieve a description of a specific frame within that video.'

    else:
        if temporal_aggregation_type == 'seq_of_f':
            class_instruct = 'Given an activity, retrieve a video description that may depict this activity.'
        else:
            class_instruct = 'Given an activity, retrieve a video frame description that may depict this activity.'
    return class_instruct


def get_description_instruct_prompt(concept, temporal_aggregation_type, is_retrieval):
    assert temporal_aggregation_type in ['middle_f', 'max_score_f', 'avg_over_f', 'seq_of_f']

    if temporal_aggregation_type == 'seq_of_f' and concept == 'objects+composition+activities':
        # use objects+composition+activities_15_words instead
        raise NotImplementedError
    if temporal_aggregation_type != 'seq_of_f' and concept == 'objects+composition+activities_15_words':
        # use objects+composition+activities instead
        raise NotImplementedError

    if is_retrieval:
        if temporal_aggregation_type == 'seq_of_f':
            instructs = {
                'objects+composition+activities_15_words': 'Given descriptions of video frames, retrieve a short description of the full video.',
                'objects': 'Given lists of objects visible on the video frames, retrieve a short video description.',
                'activities': 'Given a description of actions visible on the video frames, retrieve a short video description.',
                'verbs': 'Given a description of actions visible on the video frames, retrieve a short video description.',
            }
        else:
            instructs = {
                'objects+composition+activities': 'Given a description of a single video frame, retrieve a short description of the full video.',
                'objects': 'Given a list of objects visible on the video frame, retrieve a short video description.',
                'activities': 'Given a description of actions visible on the video frame, retrieve a short video description.',
                'verbs': 'Given a description of actions visible on the video frame, retrieve a short video description.',
            }
    else:
        if temporal_aggregation_type == 'seq_of_f':
            instructs = {
                'objects+composition+activities_15_words': 'Given descriptions of video frames, retrieve the activity depicted in this video.',
                'objects': 'Given lists of objects visible on the video frames, retrieve the activity depicted in this video.',
                'activities': 'Given a description of actions visible on the video frames, retrieve the activity depicted in this video.',
                'verbs': 'Given a description of actions visible on the video frames, retrieve the activity depicted in this video.',
            }
        else:
            instructs = {
                'objects+composition+activities': 'Given a video frame description, retrieve the activity depicted in this video.',
                'objects': 'Given a list of objects visible on the video frame, retrieve the activity depicted in this video.',
                'activities': 'Given a description of actions visible on the video frame, retrieve the activity depicted in this video.',
                'verbs': 'Given a description of actions visible on the video frame, retrieve the activity depicted in this video.',
            }
    return instructs[concept]


def construct_instruction(instruct, query):
    return f'Instruct: {instruct.strip()}\nQuery: {query.strip()}'


def aggregate_temporally(concepts_per_frame, temporal_aggregation_type):
    assert temporal_aggregation_type in ['middle_f', 'max_score_f', 'avg_over_f', 'seq_of_f']

    if temporal_aggregation_type == 'middle_f':
        middle_n = int(len(concepts_per_frame) / 2) - 1
        concepts_per_frame = [concepts_per_frame[middle_n]]

    concepts_per_frame = [x.strip() for x in concepts_per_frame]

    if temporal_aggregation_type in ['middle_f', 'max_score_f', 'avg_over_f']:
        # skip empty
        output = [x for x in concepts_per_frame if x != '']
    elif temporal_aggregation_type == 'seq_of_f':
        template = (
            "Frame #1: {}\n"
            "Frame #2: {}\n"
            "Frame #3: {}\n"
            "Frame #4: {}\n"
            "Frame #5: {}\n"
            "Frame #6: {}\n"
            "Frame #7: {}\n"
            "Frame #8: {}\n"
        )
        output = [template.format(*concepts_per_frame)]
    else:
        raise NotImplementedError
    return output


def get_pooled_text_embeddings(model, model_name, tokenize, feat_dim, texts, truncate=False,
                               batch_size=64):
    texts_all = list(itertools.chain.from_iterable(texts))
    feat_all = get_text_embeddings(model, model_name, tokenize, feat_dim, texts_all, truncate=truncate, batch_size=batch_size)

    lengths = [len(cur_texts) for cur_texts in texts]
    feats = pooling_embeddings(feat_all, lengths)
    return feats


def get_text_embeddings(model, model_name, tokenize, clip_feat_dim, in_text, truncate=False, batch_size=64):
    N = len(in_text)
    if model_name in ['clip', 'longclip']:
        in_text = tokenize(in_text, truncate=truncate).cuda()

    text_id = 0
    text_feats = np.zeros((N, clip_feat_dim), dtype=np.float32)
    for _ in tqdm.tqdm(range(int(np.ceil(N / batch_size)))):
        batch_size = min(N - text_id, batch_size)
        text_batch = in_text[text_id:text_id + batch_size]
        with torch.no_grad():
            if model_name in ['clip', 'longclip']:
                batch_feats = model.encode_text(text_batch).float()
            else:
                batch_feats = model.encode(text_batch)
                batch_feats = torch.from_numpy(batch_feats).float()
        batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
        batch_feats = np.float32(batch_feats.cpu())
        text_feats[text_id:text_id + batch_size, :] = batch_feats
        text_id += batch_size
    return text_feats


def normalize(prediction_feat):
    return prediction_feat / np.linalg.norm(prediction_feat, axis=1, keepdims=True)