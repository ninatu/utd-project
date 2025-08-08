import pickle
import re

import numpy as np
import os
import json


def pooling_embeddings(list_of_features, list_of_lengths, pooling='mean'):
    offset = 0
    feats = []

    for n in list_of_lengths:
        if n == 0:
            feat = np.zeros(list_of_features.shape[1])
        else:
            feat = list_of_features[offset: offset + n]
            if pooling == 'mean':
                feat = feat.mean(0)
            elif pooling == 'max':
                feat = feat.max(0)
            else:
                raise NotImplementedError

        feats.append(feat)
        offset += n
    feats = np.stack(feats, axis=0)
    return feats


def load_descriptions(root):
    if root.endswith('.json'):
        with open(root, 'r') as fin:
            return json.load(fin)
    else:
        with open(root, 'rb') as fin:
            return pickle.load(fin)
