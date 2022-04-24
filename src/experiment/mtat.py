from pathlib import Path

import numpy as np
from scipy import sparse as sp
import h5py

from tqdm import tqdm

from bibim.data import MVVarSeqData

from .common import (MultLabelClfTarget,
                     TestDataset,
                     process_loudness,
                     classification_test,
                     _macro_aucroc_scoring_safe,
                     MODEL_MAP)


JORDI_TAG50 = [
    ("guitar", 1), ("classical", 2), ("slow", 3), ("techno", 4),
    ("strings", 5), ("drums", 6), ("electronic", 7), ("rock", 8), ("fast", 9),
    ("piano", 10), ("ambient", 11), ("beat", 12), ("violin", 13), ("vocal", 14),
    ("synth", 15), ("female", 16), ("indian", 17), ("opera", 18), ("male", 19),
    ("singing", 20), ("vocals", 21), ("no vocals", 22), ("harpsichord", 23),
    ("loud", 24), ("quiet", 25), ("flute", 26), ("woman", 27), ("male vocal", 28),
    ("no vocal", 29), ("pop", 30), ("soft", 31), ("sitar", 32), ("solo", 33),
    ("man", 34), ("classic", 35), ("choir", 36), ("voice", 37), ("new age", 38),
    ("dance", 39), ("male voice", 40), ("female vocal", 41), ("beats", 42),
    ("harp", 43), ("cello", 44), ("no voice", 45), ("weird", 46), ("country", 47),
    ("metal", 48), ("female voice", 49), ("choral", 50)
]


def load_mtat(
    h5_fn: str,
    split_fn: str,
    tag50: bool = True,
) -> tuple[TestDataset, h5py.File]:
    """
    """
    id2split = {}
    with Path(split_fn).open() as fp:
        for line in fp:
            i, split_name = line.replace('\n','').split(',')
            id2split[int(i)] = split_name

    hf = h5py.File(h5_fn)
    n_samples = hf['indptr'].shape[0] - 1
    indptr = hf['indptr'][:]

    loudness = hf['data'][:, 0]

    # ids = [f'{i:d}' for i in hf['ids'][:]]
    tag2id = {t.decode():i for i, t
              in enumerate(hf['annotations']['tags'][:])}
    jordi50_ids = [tag2id[t] for t, i in JORDI_TAG50]

    # compute loudness feature (that we'll discard for HDPGMM)
    black_list = []
    rows = []
    splits = []
    indptr_ = [0]
    ids = []
    for j in range(len(indptr) - 1):
        id_j = hf['ids'][j]
        j0, j1 = indptr[j], indptr[j+1]
        if j1 == j0 or id_j not in id2split:
            black_list.append(j)
            print(f'[Warning] no frame found for {j:d}th sample!')
            continue

        ids.append(id_j)

        # compute some basic stats
        loudness_j = loudness[j0:j1]
        rows.append(process_loudness(loudness_j))
        indptr_.append(indptr_[-1] + j1 - j0)
        splits.append(id2split[id_j])

    splits = np.array(splits)
    loudness_feat = np.array(rows)
    loudness_feat = (
        (loudness_feat - np.mean(loudness_feat, axis=0)[None])
        / np.std(loudness_feat, axis=0)[None]
    )

    annot = hf['annotations']
    targets = sp.csr_matrix(
        (annot['data'][:],
         annot['indices'][:],
         annot['indptr'][:]),
        shape = (hf['ids'].shape[0],
                 annot['tags'].shape[0])
    )
    if tag50:
        targets = targets[:, jordi50_ids]
        idx2tags = [t for t, i in JORDI_TAG50]
    else:
        idx2tags = [t.decode() for t in annot['tags'][:]]

    black_list = set(black_list)
    to_keep = np.array([i for i in range(n_samples) if i not in black_list])
    targets = targets[to_keep]
    assert targets.shape[0] == (n_samples - len(black_list))

    # wrap the raw dataset into the 
    dataset = MVVarSeqData(np.array(indptr_), hf['data'], ids)
    assert len(dataset.ids) == (len(dataset.indptr) - 1)

    # build target
    target = MultLabelClfTarget(labels=idx2tags, label_map=targets)

    return TestDataset(dataset, loudness_feat, target, splits), hf


def run_experiment(
    model_class: str,
    model_fn: str,
    mtat_fn: str,
    mtat_split_fn: str,
    n_iters: int = 5,
    batch_size: int = 1024,
    n_jobs: int = 1,
    accelerator: str = 'cpu',
    verbose: bool = False
) -> list[float]:
    """
    """
    dataset, hf = load_mtat(
        mtat_fn,
        mtat_split_fn,
    )
    model = MODEL_MAP[model_class].load(model_fn)
    model.n_jobs = n_jobs  # only relevant with HPDGMM for now
    model.batch_size = batch_size
    config = model.get_config()

    accs = []
    with tqdm(total=n_iters, ncols=80, disable=not verbose) as prog:
        for _ in range(n_iters):
            acc = classification_test(model, dataset,
                                      eval_metric=_macro_aucroc_scoring_safe,
                                      n_jobs=n_jobs,
                                      accelerator=accelerator)
            accs.append(acc)
            prog.update()

    hf.close()
    result = config.copy()
    result['task'] = 'magnatagatune'
    result['performance'] = accs
    result['performance_metric'] = 'rocauc_score_macro'
    return result
