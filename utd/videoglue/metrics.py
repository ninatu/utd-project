import numpy as np
import scipy
import scipy.stats
import itertools


def compute_retrieval_metrics_per_query(sims, from_ids, to_ids, break_ties='averaging'):
    num_queries, num_keys = sims.shape
    assert num_queries == len(from_ids)
    assert num_keys == len(to_ids)

    sx = np.sort(-sims, axis=1)
    gt = np.array([-sims[i, to_ids.index(id)] for i, id in enumerate(from_ids)])
    gt = gt[:, np.newaxis]
    diff = sx - gt
    if break_ties == 'optimistically':
        ind = np.argmax(diff == 0, axis=1)
    elif break_ties == 'averaging':
        locs = np.argwhere(diff == 0)
        grouped_locs = [list(values) for n_row, values in itertools.groupby(locs, key=lambda x: x[0])]
        ind = [np.mean(list(map(lambda x: x[1], locs))) for locs in grouped_locs]
        ind = np.array(ind)
    else:
        raise NotImplementedError
    output = {}
    output['rank'] = ind
    output["R1"] = (ind == 0).astype(float)
    output["R5"] = (ind < 5).astype(float)
    output["R10"] = (ind < 10).astype(float)
    output["R50"] = (ind < 50).astype(float)

    return output
