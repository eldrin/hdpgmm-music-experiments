from pathlib import Path
import pickle as pkl

import numpy as np
from scipy import sparse as sp
import h5py

from tqdm import tqdm

from hdpgmm.data import HDFMultiVarSeqDataset

from .common import (RecSysInteractionTarget,
                     TestDataset,
                     load_model,
                     recommendation_test,
                     process_loudness)


def _load_interaction(
    triplet_fn: str,
    song2track_fn: str
) -> tuple[sp.csr_matrix,
           list[str],
           list[str]]:
    """
    """
    # load song to track map
    with Path(song2track_fn).open('rb') as fp:
        song2track = pkl.load(fp)

    # load the triplet
    I, J, V = [], [], []
    users, user_list = {}, []
    items, item_list = {}, []
    with Path(triplet_fn).open('r') as fp:
        for line in fp:
            user, song, count = line.replace('\n', '').split(',')
            track = song2track[song]

            if user not in users:
                users[user] = len(users)
                user_list.append(user)

            if track not in items:
                items[track] = len(items)
                item_list.append(track)

            I.append(users[user])
            J.append(items[track])
            V.append(int(count))
    mat = sp.coo_matrix((V, (I, J)), shape=(len(users), len(items))).tocsr()

    return mat, user_list, item_list


def load_echonest(
    h5_fn: str,
    *args, **kwargs
) -> TestDataset:
    """
    """
    # load audio feature data
    hf = h5py.File(h5_fn)
    indptr = hf['indptr'][:]
    # loudness = hf['data'][:, 0]
    ids = [i.decode() for i in hf['ids'][:]]

    # compute loudness feature (that we'll discard for HDPGMM)
    rows = []
    for j in range(len(indptr) - 1):
        j0, j1 = indptr[j], indptr[j+1]
        n_frames = j1 - j0

        # compute some basic stats
        # loudness_j = loudness[j0:j1]
        loudness_j = hf['data'][j0:j1, 0]
        rows.append(process_loudness(loudness_j))

    loudness_feat = np.array(rows)
    loudness_feat = (
        (loudness_feat - np.mean(loudness_feat, axis=0)[None])
        / np.std(loudness_feat, axis=0)[None]
    )

    # load target
    users = [u.decode() for u in hf['interaction/userlist'][:]]
    items = [i.decode() for i in hf['interaction/itemlist'][:]]
    mat = sp.csr_matrix(
        (hf['interaction/data'][:],
         hf['interaction/indices'][:],
         hf['interaction/indptr'][:]),
        shape=(len(users), len(items))
    )
    hf.close()

    # wrap the raw dataset into the
    dataset = HDFMultiVarSeqDataset(h5_fn)

    # wrap the data
    target = RecSysInteractionTarget(mat, users, items)

    return TestDataset(dataset, loudness_feat, target)


def run_experiment(
    model_class: str,
    model_fn: str,
    lfm1k_h5_fn: str,
    n_iters: int = 5,
    n_jobs: int = 1,
    batch_size: int = 1024,
    top_k: int = 500,
    valid_user_ratio: float = 0.1,
    test_user_ratio: float = 0.1,
    similarity: str = 'cosine',
    device: str = 'cpu',
    verbose: bool = False
) -> list[float]:
    """
    """
    dataset = load_echonest(lfm1k_h5_fn)
    model = load_model(model_fn, model_class, dataset,
                       batch_size = batch_size)
    config = model.get_config()

    accs = []
    with tqdm(total=n_iters, ncols=80, disable=not verbose) as prog:
        for n in range(n_iters):
            acc = recommendation_test(model, dataset,
                                      n_splits=5,
                                      similarity=similarity,
                                      device=device,
                                      valid_user_ratio=valid_user_ratio,
                                      test_user_ratio=test_user_ratio,
                                      sample_user=batch_size,
                                      top_k=top_k)
            accs.append(np.mean(acc))
            prog.update()

    result = config.copy()
    result['task'] = 'echonest'
    result['performance'] = accs
    result['performance_metric'] = f'ndcg@{top_k:d}'
    return result
