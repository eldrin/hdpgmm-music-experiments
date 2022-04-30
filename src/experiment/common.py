from typing import Union, Optional, Callable
from dataclasses import dataclass
from functools import partial
import pickle as pkl

import numpy as np
from numpy import typing as npt
import numba as nb
from numba.typed import List as TypedList

from scipy.stats import uniform, loguniform
from scipy import sparse as sp

from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score,
                             average_precision_score)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
# from skopt import BayesSearchCV

import torch
import h5py
import librosa

from bibim.hdp import gaussian as hdpgmm
from bibim.data import MVVarSeqData

from ..models import (FeatureLearner,
                      HDPGMM, VQCodebook, G1, PreComputedFeature)
from .custom_modules import LitSKLogisticRegression
from .itemknn import cos_dist, js_div, build_knn_sim_csr


percentile25 = partial(np.percentile, q=25)
percentile75 = partial(np.percentile, q=75)


MODEL_MAP = {
    'hdpgmm': HDPGMM,
    'vqcodebook': VQCodebook,
    'G1': G1,
    'precomputed': PreComputedFeature
}

SIM_FUNC_MAP = {
    'jensenshannon': lambda p, q: -js_div(p.exp(), q.exp()),
    'cosine': lambda p, q: 1. - cos_dist(p, q)
}


@nb.jit(cache=True, fastmath=True)
def ndcg(
    actual: set[int],
    predicted: list[int],
    k: int=10
) -> float:
    """for binary relavance

    NOTE: although it's recommended to use numba.typed.List,
          reflected list still is much faster due to the
          issues around the implementation of typed List
          (https://github.com/numba/numba/issues/4584)

    TODO: using cfunc would be slightly faster
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    n_actual = len(actual)
    if n_actual == 0:
        return 0.

    dcg = 0.
    idcg = 0.
    found = set()
    for i, p in enumerate(predicted):
        if p in actual and p not in found:
            found.add(p)
            dcg += 1. / np.log2(i + 2.)
        if i < n_actual:
            idcg += 1. / np.log2(i + 2.)

    return dcg / idcg


@nb.njit(cache=True)
def ndcg_npy(actual, predicted, k=10):
    """ for binary relavance """
    if len(predicted) > k:
        predicted = predicted[:k]

    dcg = 0.
    idcg = 0.
    for i, p in enumerate(predicted):
        if np.any(actual == p) and np.all(predicted[:i] != p):
            dcg += 1. / np.log2(i + 2.)
        if i < len(actual):
            idcg += 1. / np.log2(i + 2.)

    if len(actual) == 0:
        return 0.

    return dcg / idcg


@nb.njit(parallel=True, nogil=True, fastmath=True)
def _ndcg_batch(
    pred_batch: npt.ArrayLike,
    test_indptr: npt.ArrayLike,
    test_indices: npt.ArrayLike,
    top_k: int
) -> npt.ArrayLike:
    """
    """
    ndcgs = np.empty((pred_batch.shape[0],))
    ndcgs[:] = np.nan
    rnd_idx = np.random.permutation(pred_batch.shape[0])
    for i in nb.prange(pred_batch.shape[0]):
        u = rnd_idx[i]
        u0, u1 = test_indptr[u], test_indptr[u + 1]
        if u1 == u0:
            continue

        # predict and get the ranked list
        pred = pred_batch[u]
        true = test_indices[u0:u1]
        ndcgs[u] = ndcg_npy(true, pred, top_k)

    return ndcgs


def ndcg_batch(
    S: npt.ArrayLike,
    test: sp.csr_matrix,
    top_k: int
) -> npt.ArrayLike:
    """
    S: top-k similar items' matrix
    heldout_batch: sparse matrix contains the binarized
                   heldout interaction records for
                   the mini-batch of users
    """
    # first we compute the predictions before going into the loop
    batch_users = S.shape[0]
    idx_topk_part = np.argpartition(-S, top_k, axis=1)[:, :top_k]
    topk_part = S[np.arange(batch_users)[:, None], idx_topk_part]
    idx_part = np.argsort(-topk_part, axis=1)
    pred_batch = idx_topk_part[np.arange(batch_users)[:, None], idx_part]
    return _ndcg_batch(pred_batch, test.indptr, test.indices, top_k)


@dataclass
class PredictionTarget:
    labels: list[str]


@dataclass
class MultClassClfTarget(PredictionTarget):
    label_map: npt.ArrayLike  # list of label indices (int) per sample


@dataclass
class MultLabelClfTarget(PredictionTarget):
    label_map: sp.csr_matrix  # 2D sparse matrix (CSR) mapping sample to labels


@dataclass
class RecSysInteractionTarget:
    interaction: sp.csr_matrix
    users: list[str]
    items: list[str]


@dataclass
class TestDataset:
    data: MVVarSeqData
    loudness: npt.ArrayLike
    target: Union[PredictionTarget, RecSysInteractionTarget]
    splits: Optional[npt.ArrayLike] = None


def process_loudness(
    loudness: npt.ArrayLike
) -> npt.ArrayLike:
    """
    """
    return np.asarray([
        stat_fn(loudness)
        for stat_fn
        in [np.mean, np.std, percentile25, np.median, percentile75]
    ])


def load_model(
    model_fn: str
) -> hdpgmm.HDPGMM:
    """
    """
    with open(model_fn, 'rb') as fp:
        model = pkl.load(fp)
    return model


def infer_documents(
    model: hdpgmm.HDPGMM,
    dataset: MVVarSeqData,
    n_max_inner_update: int = 100,
    e_step_tol: float = 1e-4
) -> tuple[npt.ArrayLike,  # mean responsibility
           npt.ArrayLike]: # mean likelihood
    """
    """
    (
        a, b, eq_pi, w, prior, lik
    ) = hdpgmm.infer_documents(dataset, model,
                               n_max_inner_update=n_max_inner_update,
                               e_step_tol=e_step_tol)
    return prior, lik


def process_feature(
    model: FeatureLearner,
    dataset: TestDataset,
    loudness_cols: bool = False,
    eps: float = 10. * np.finfo('float64').eps
) -> tuple[npt.ArrayLike,
           npt.ArrayLike]:
    """
    """
    # extract feature
    if isinstance(model, (HDPGMM, VQCodebook)):
        lik_ = model.extract(dataset.data)

        # preprocessing (drop out components having no activation)
        X = np.log(np.maximum(lik_, eps))

    else:
        X = model.extract(dataset.data)

    # combine loudness columns to the log-likelihoods
    if loudness_cols:
        X = np.c_[X, dataset.loudness]

    if not isinstance(dataset.target, RecSysInteractionTarget):
        y = dataset.target.label_map

        # most of models takes dense targets, but some
        # doesn't work with the sparse target matrix
        if isinstance(dataset.target, MultLabelClfTarget):
            y = np.asarray(y.todense())
    else:
        y = dataset.target.interaction

    return X, y


def _macro_aucroc_scoring_safe(model, x_test, y_test):
    """
    """
    y_hat = model.predict_log_proba(x_test)
    good_cols = y_test.sum(0) > 0
    return roc_auc_score(y_test[:, good_cols],
                         y_hat[:, good_cols],
                         average='macro')


def _macro_average_precision_scoring_safe(model, x_test, y_test):
    """
    """
    y_hat = model.predict_log_proba(x_test)
    good_cols = y_test.sum(0) > 0
    return average_precision_score(y_test[:, good_cols],
                                   y_hat[:, good_cols],
                                   average='macro')


def _accuracy_scoring(model, x_test, y_test):
    """
    """
    y_pred = model.predict(x_test)
    return accuracy_score(y_test, y_pred)


def _macro_f1_scoring(model, x_test, y_test):
    """
    """
    y_pred = model.predict(x_test)
    return f1_score(y_test, y_pred, average='macro')


def score_clf(
    eval_metric: Union[Callable, dict[str, Callable]],
    model: object,  # fitted meta-estimator, such as RandomizedSearchCV
    X: npt.ArrayLike,
    y: npt.ArrayLike
) -> dict[str, float]:
    """
    """
    # fit with full training data with found optimal hyper-parameter
    if isinstance(eval_metric, dict):
        # compute all scores given in the multi-score dictionary
        score = {
            name: func(model, X, y)
            for name, func in eval_metric.items()
        }
    else:
        score = {'score': eval_metric(model, X, y)}
    return score


def classification_test(
    model: FeatureLearner,
    dataset: TestDataset,
    n_splits: int = 5,
    n_rnd_srch_iter: int = 32,
    n_jobs: int = 4,
    eval_metric: Union[Callable,
                       dict[str, Callable]] = _accuracy_scoring,
    accelerator: str = 'cpu'  # only relevant for Multilabel Classification
) -> list[float]:
    """
    """
    assert accelerator in {'cpu', 'gpu'}

    # process feature
    X, y = process_feature(model, dataset)

    # initialize model
    if isinstance(dataset.target, MultLabelClfTarget):
        est = Pipeline([('z_score', StandardScaler()),
                        ('lr', LitSKLogisticRegression(accelerator=accelerator,
                                                       loss='bin_xent',
                                                       num_workers=0))])
        dist = dict(lr__alpha=loguniform(1e-4, 1e+4),
                    lr__learning_rate=loguniform(1e-7, 1e-1),
                    lr__batch_size=[256, 512, 1024, 2048],
                    lr__max_iter=[250, 500, 1000, 2000])
        # dist = dict(lr__alpha=(1e-4, 1e+4, 'log-uniform'),
        #             lr__learning_rate=(1e-7, 1e+1, 'log-uniform'),
        #             lr__batch_size=[256, 512, 1024, 2048],
        #             lr__max_iter=[250, 500, 1000, 2000])

    elif isinstance(dataset.target, MultClassClfTarget):
        est = Pipeline([('z_score', StandardScaler()),
                        ('lr', LogisticRegression(max_iter=20000))])
        dist = dict(lr__C=loguniform(1e-3, 1e+3))
        # dist = dict(lr__C = (1e-3, 1e+3, 'log-uniform'))

    else:
        raise ValueError('[Error] task not known!')

    # setup learning / testing
    accs = []
    if dataset.splits is not None:
        train_ix = np.where(dataset.splits == 'train')[0]
        valid_ix = np.where(dataset.splits == 'valid')[0]
        test_ix = np.where(dataset.splits == 'test')[0]
        train_valid_idx = np.r_[train_ix, valid_ix]
        train_bound = len(train_ix)
        cv_ = ((np.arange(train_bound),
                np.arange(train_bound, len(train_valid_idx))),)

        # fit the model
        if isinstance(eval_metric, dict):
            # TODO: we probably should expose this as
            #       user-configurable parameter
            refit = next(iter(eval_metric.keys()))
        else:
            refit = True
        clf = RandomizedSearchCV(est, dist,
                                 scoring=eval_metric,
                                 refit=refit,
                                 n_jobs=n_jobs,
                                 n_iter=n_rnd_srch_iter,
                                 cv=cv_)
        # clf = BayesSearchCV(est, dist,
        #                     scoring=eval_metric,
        #                     refit=refit,
        #                     n_iter=n_rnd_srch_iter,
        #                     cv=cv_)

        # find best model
        search = clf.fit(X[train_valid_idx], y[train_valid_idx])

        # compute test performance
        acc = score_clf(eval_metric, search, X[test_ix], y[test_ix])
        accs.append(acc)

    else:
        rs = ShuffleSplit(n_splits = n_splits, test_size=.2)
        for train_idx, test_idx in rs.split(X):
            clf = RandomizedSearchCV(est, dist,
                                     scoring=eval_metric,
                                     refit=True,
                                     n_iter=n_rnd_srch_iter)
            # clf = BayesSearchCV(est, dist,
            #                     scoring=eval_metric,
            #                     refit=refit,
            #                     n_iter=n_rnd_srch_iter)
            search = clf.fit(X[train_idx], y[train_idx])

            acc = score_clf(eval_metric, search, X[test_ix], y[test_ix])
            accs.append(acc)

    return accs


# @nb.njit
def __split_data(
    indptr: npt.ArrayLike,
    indices: npt.ArrayLike,
    data: npt.ArrayLike,
    train_ratio: float
) -> tuple[npt.ArrayLike,
           npt.ArrayLike,
           npt.ArrayLike,
           npt.ArrayLike,
           npt.ArrayLike,
           npt.ArrayLike]:
    """
    """
    # initialize the output containers
    indptr_tr = [0]
    indices_tr = []
    data_tr = []
    indptr_ts = [0]
    indices_ts = []
    data_ts = []

    n_users = len(indptr) - 1
    n_inds_tr = 0
    n_inds_ts = 0
    for u in range(n_users):
        u0, u1 = indptr[u], indptr[u+1]
        if u1 == u0:
            indptr_tr.append(indptr_tr[-1])
            indptr_ts.append(indptr_ts[-1])
            continue

        n_records = u1 - u0
        inds = indices[u0:u1]
        vals = data[u0:u1]

        # split the data
        _rnd_idx = np.random.permutation(n_records)
        train_bnd = max(int(train_ratio * n_records), 1)

        n_train = train_bnd
        n_test = n_records - train_bnd

        indptr_tr.append(indptr_tr[-1] + n_train)
        indptr_ts.append(indptr_ts[-1] + n_test)

        indices_tr.append(inds[_rnd_idx[:train_bnd]])
        indices_ts.append(inds[_rnd_idx[train_bnd:]])

        data_tr.append(vals[_rnd_idx[:train_bnd]])
        data_ts.append(vals[_rnd_idx[train_bnd:]])

    # post-process
    indptr_tr = np.array(indptr_tr)
    indptr_ts = np.array(indptr_ts)

    indices_tr = np.concatenate(indices_tr)
    indices_ts = np.concatenate(indices_ts)

    data_tr = np.concatenate(data_tr)
    data_ts = np.concatenate(data_ts)

    return (
        indptr_tr, indices_tr, data_tr,
        indptr_ts, indices_ts, data_ts
    )


def _split_data(
    total_interaction: sp.csr_matrix,
    per_user_train_ratio: float = 0.7,
    valid_user_ratio: float = 0.1,
    test_user_ratio: float = 0.1
) -> tuple[tuple[sp.csr_matrix,  # train
                 sp.csr_matrix], # valid_test
           tuple[int,  # train user index bound
                 int]]: # valid user index bound
    """
    """
    # sample valid / test users
    n_users, n_items = total_interaction.shape
    rnd_idx = np.random.permutation(n_users)
    n_valid = max(int(n_users * valid_user_ratio), 1)
    n_test = max(int(n_users * test_user_ratio), 1)
    train_bnd = n_users - n_valid - n_test

    user_splits = {}
    user_splits['train'] = rnd_idx[:train_bnd]
    user_splits['valid'] = rnd_idx[train_bnd:train_bnd + n_valid]
    user_splits['test'] = rnd_idx[train_bnd + n_valid:]

    # split records "per user" for valid/test
    valid_test = {}
    for split in ['valid', 'test']:
        user_indices = user_splits[split]
        split_rows = total_interaction[user_indices]

        (indptr_tr,
         indices_tr,
         data_tr,
         indptr_ts,
         indices_ts,
         data_ts) = __split_data(split_rows.indptr,
                                 split_rows.indices,
                                 split_rows.data,
                                 train_ratio = per_user_train_ratio)

        # wrap the matrices
        valid_test[split] = {}
        valid_test[split]['train'] = sp.csr_matrix(
            (data_tr, indices_tr, indptr_tr),
            shape=(len(user_indices), n_items)
        )
        valid_test[split]['test'] = sp.csr_matrix(
            (data_ts, indices_ts, indptr_ts),
            shape=(len(user_indices), n_items)
        )

    # wrap up
    train = sp.vstack([total_interaction[user_splits['train']],
                       valid_test['valid']['train'],
                       valid_test['test']['train']])
    valid_test = sp.vstack([total_interaction[user_splits['train']],
                            valid_test['valid']['test'],
                            valid_test['test']['test']])
    return (
        (train, valid_test),
        (train_bnd, train_bnd + n_valid)
    )


def _test_per_k(
    train: sp.csr_matrix,
    valid_test: sp.csr_matrix,
    M: npt.ArrayLike,
    train_bound: int,
    valid_bound: int,
    top_k: int = 500,
    is_valid: bool = True,
    binarize: bool = True
) -> float:
    """
    """
    # get range of indices to be tested
    if is_valid:
        testing_targets = np.arange(train_bound, valid_bound)
        n_testing_users = len(testing_targets)
    else:
        testing_targets = np.arange(valid_bound, train.shape[0])
        n_testing_users = len(testing_targets)

    # compute the K-similar items matrix (n_items, n_items)
    test_train = train[testing_targets].copy()
    test_test = valid_test[testing_targets].copy()

    if binarize:
        test_train.data[:] = 1.
        test_test.data[:] = 1.

    S = (test_train @ M).todense()
    accs = ndcg_batch(S, test_test, top_k)

    return np.nanmean(accs)


def keep_top_k(
    item_sim_mat: sp.csr_matrix,
    k: int
) -> sp.csr_matrix:
    """
    """
    indptr = [0]
    indices = []
    data = []
    for j in range(item_sim_mat.shape[0]):
        j0, j1 = item_sim_mat.indptr[j], item_sim_mat.indptr[j+1]
        if j1 == j0:
            indptr.append(indptr[-1])
            continue

        # slice the data
        ind = item_sim_mat.indices[j0:j1]
        dat = item_sim_mat.data[j0:j1]

        # take top-k given k and add to the buffers
        ind_topk = np.argpartition(-dat, kth=k)[:k]

        indptr.append(indptr[-1] + k)
        indices.append(ind[ind_topk])
        data.append(dat[ind_topk])

    # build a new sparse matrix and return
    indptr = np.array(indptr)
    indices = np.concatenate(indices)
    data = np.concatenate(data)
    return sp.csr_matrix((data, indices, indptr), shape=item_sim_mat.shape)


def _knn_recsys_test(
    feature: torch.Tensor,
    total_interaction: sp.csr_matrix,
    item_sim_mat: sp.csr_matrix,
    train_ratio: float = 0.7,
    valid_user_ratio: float = 0.1,
    test_user_ratio: float = 0.1,
    k_range: list[int] = [16, 32, 64, 128, 256, 512],
    top_k: int = 500,
    binarize: bool = True
) -> float:
    """
    we expect L2-normalized feature, to approximate
    cosine similarity between items via a simple rank-restricted SVD
    """
    # split the data into train/test
    splits = _split_data(total_interaction,
                         valid_user_ratio=valid_user_ratio,
                         test_user_ratio=test_user_ratio)
    (train, valid_test), (trn_bnd, val_bnd) = splits

    # sweep over the candidate range of `k` to find the best
    Ms = {}
    valid_accs = []
    for k in k_range:
        if k > train.shape[1]:
            break

        if k >= max(k_range):
            Ms[k] = item_sim_mat
        else:
            Ms[k] = keep_top_k(item_sim_mat, k)

        valid_acc = _test_per_k(train, valid_test, Ms[k],
                                trn_bnd, val_bnd, top_k,
                                is_valid=True)
        valid_accs.append(valid_acc)

    # pick the best k and compute error based on it
    best_k = k_range[np.argmax(valid_accs)]
    test_acc = _test_per_k(train, valid_test, Ms[best_k],
                           trn_bnd, val_bnd, top_k,
                           is_valid=False, binarize=binarize)
    return test_acc, best_k


def recommendation_test(
    model: FeatureLearner,
    dataset: TestDataset,
    top_k: int = 500,
    train_ratio: float = 0.7,
    valid_user_ratio: float = 0.1,
    test_user_ratio: float = 0.1,
    k_range: list[int] = [16, 32, 64, 128, 256, 512],
    n_splits: int = 5,
    sample_user: Optional[Union[float, int]] = 200,
    similarity: str = 'cosine',
    device: str = 'cpu'
) -> list[float]:
    """
    feature:
        the numpy array contains the item feature having shape of (n_items, n_dim)
        it is expected to the same order as the interaction matrix

    interaction:
        it contains the interaction matrix / user list / item list

    top_k:
        the cutoff value to compute the performance measure

    sample_user:
        if None -> no sampling and compute to all
        if float[0, 1] -> sample the sub-population of users in fraction
        if int ( > 0) -> sample the number of users by given positive integer
    """
    n_users = len(dataset.target.users)

    # checking the similarity function given
    assert similarity in SIM_FUNC_MAP

    # checking the number of user-sampling
    assert sample_user is None or isinstance(sample_user, (float, int))
    if isinstance(sample_user, float):
        assert sample_user > 0. and sample_user <= 1.
        n_sample = max(int(sample_user * n_users), 1)
    elif isinstance(sample_user, int):
        assert sample_user >= 1
        n_sample = sample_user
    else:  # sample_user == None
        n_sample = n_users

    # process feature
    print('process feature!')
    X, y = process_feature(model, dataset)
    X = torch.as_tensor(X, device=device)

    # compute item sim matrix (truncated on the largest K)
    print('building sim matrix!')
    M = build_knn_sim_csr(X, max(k_range), SIM_FUNC_MAP[similarity])

    accs = []
    for i in range(n_splits):
        print(f'knn test {i:d}th!')
        acc, best_k = _knn_recsys_test(X, y, M,
                                       train_ratio,
                                       valid_user_ratio,
                                       test_user_ratio,
                                       k_range,
                                       top_k)
        accs.append(acc)
    return accs
