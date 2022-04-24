from pathlib import Path
from functools import partial

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import h5py

from tqdm import tqdm

from bibim.data import MVVarSeqData

from .common import (MultClassClfTarget,
                     TestDataset,
                     process_loudness,
                     classification_test,
                     MODEL_MAP)


def load_gtzan(
    h5_fn: str,
    split_fn: str,
    exclude_1st_dim: bool = False,
    exclude_chroma_dims: bool = False
) -> TestDataset:
    """
    """
    gtzan_artist_split = pd.read_csv(split_fn,
                                     header=None,
                                     index_col=None,
                                     sep='\t')
    id2split = {
        p.split('/')[-1]: s
        for p, s
        in gtzan_artist_split[[1, 2]].values
    }

    data_ = []
    indptr_ = [0]
    with h5py.File(h5_fn) as hf:
        indptr = hf['indptr'][:]
        data = hf['data'][:]
        ids = [i.decode() for i in hf['ids'][:]]

        # compute loudness feature (that we'll discard for HDPGMM)
        rows = []
        splits = []
        for j in range(len(indptr) - 1):
            j0, j1 = indptr[j], indptr[j+1]

            # compute some basic stats
            loudness_j = data[j0:j1, 0]
            rows.append(process_loudness(loudness_j))

            splits.append(id2split[hf['ids'][j].decode()])

            x = data[j0:j1]
            if exclude_1st_dim:
                x = x[:, 1:]
            if exclude_chroma_dims:
                x = x[:, :-12]

            data_.append(x)
            indptr_.append(indptr_[-1] + data_[-1].shape[0])

        splits = np.array(splits)
        loudness = np.array(rows)
        loudness = (loudness - np.mean(loudness)) / np.std(loudness)

        targets = np.array([g.decode() for g in hf['targets'][:]])

    # wrap the raw dataset into the 
    dataset = MVVarSeqData(np.array(indptr_), np.concatenate(data_), ids)

    # build target
    idx2genre = list(sorted(set(targets)))
    genre2idx = {g:i for i, g in enumerate(idx2genre)}
    genres = np.array([genre2idx[t] for t in targets])
    target = MultClassClfTarget(labels = idx2genre, label_map = genres)

    return TestDataset(dataset, loudness, target, splits)


def run_experiment(
    model_class: str,
    model_fn: str,
    gtzan_fn: str,
    gtzan_split_fn: str,
    exclude_1st_dim: bool = False,
    exclude_chroma_dims: bool = False,
    n_iters: int = 5,
    batch_size: int = 1024,
    n_jobs: int = 1,
    verbose: bool = False
) -> list[float]:
    """
    """
    dataset = load_gtzan(
        gtzan_fn,
        gtzan_split_fn,
        exclude_1st_dim = exclude_1st_dim,
        exclude_chroma_dims = exclude_chroma_dims
    )
    model = MODEL_MAP[model_class].load(model_fn)
    model.n_jobs = n_jobs  # only relevant with HPDGMM for now
    model.batch_size = batch_size
    config = model.get_config()

    accs = []
    metric = partial(f1_score, average='macro')
    with tqdm(total=n_iters, ncols=80, disable=not verbose) as prog:
        for _ in range(n_iters):
            acc = classification_test(model, dataset,
                                      n_jobs=n_jobs,
                                      eval_metric=metric)
            accs.append(np.mean(acc))
            prog.update()

    result = config.copy()
    result['task'] = 'gtzan'
    result['performance'] = accs
    result['performance_metric'] = 'f1_score_macro'
    return result
