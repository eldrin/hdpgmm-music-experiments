from typing import Union

import numpy as np
import numpy.typing as npt
import numba as nb
import h5py

from tqdm import tqdm

from bibim.data import MVVarSeqData


def _slice(data: MVVarSeqData, start: int, end: int) -> MVVarSeqData:
    """
    # This should be implemented on bibim's side in the future
    """
    assert start < data.num_docs
    assert start < end
    indptr_ = data.indptr[start:end+1] - data.indptr[start]
    data_ = data.data[data.indptr[start]:data.indptr[end]]
    ids_ = data.ids[start:end]
    return MVVarSeqData(indptr_, data_, ids_)


def slice_csr_like(
    indptr: npt.ArrayLike,
    data: h5py.Dataset,
    indices: npt.ArrayLike,
    exclude_1st_dim: bool = False,
    verbose: bool = False
) -> MVVarSeqData:
    """
    """
    n_tokens = 0
    indptr_slc = [0]
    for i in indices:
        j0, j1 = indptr[i], indptr[i+1]
        indptr_slc.append(indptr_slc[-1] + j1 - j0)
        n_tokens += j1 - j0
    indptr_slc = np.array(indptr_slc)

    D = data.shape[-1] if not exclude_1st_dim else data.shape[-1] - 1
    data_slc = np.empty((n_tokens, D), dtype=np.float32)
    last_idx = 0
    with tqdm(total=len(indices), ncols=80, disable=not verbose) as prog:
        for i in indices:
            j0, j1 = indptr[i], indptr[i+1]
            x = data[j0:j1] if not exclude_1st_dim else data[j0:j1, 1:]
            data_slc[last_idx:last_idx + j1 - j0] = x
            last_idx = last_idx + j1 - j0
            prog.update()

    return MVVarSeqData(indptr=indptr_slc, data=data_slc)


@nb.njit(nogil=True, parallel=True, fastmath=True, boundscheck=False)
def mvnorm_standardize(X, mean, prec_chol):
    """
    """
    for i in nb.prange(X.shape[0]):
        X[i] = (X[i] - mean) @ prec_chol


def load_dataset(
    hdf_fn: str,
    split: bool = True,
    train_ratio: float = 0.9,
    normalization: bool = True,
    exclude_1st_dim: bool = True
) -> Union[tuple[MVVarSeqData,
                 list[str],
                 dict[str, npt.ArrayLike]],
           tuple[dict[str, tuple[MVVarSeqData, list[str]]],
                 dict[str, npt.ArrayLike]]]:
    """
    """
    hf = h5py.File(hdf_fn)

    if split:
        # slice data in training and test

        # load the file to the memory
        # TODO: this might not necessary, but let's keep it simple for now
        ids = hf['ids'][:]
        indptr = hf['indptr'][:]
        data = hf['data']

        n_docs = ids.shape[0]
        rnd_idx = np.random.permutation(n_docs)
        train_bound = int(train_ratio * n_docs)
        tr_idx = rnd_idx[:train_bound]
        ts_idx = rnd_idx[train_bound:]

        Xtr = slice_csr_like(indptr, data, tr_idx, exclude_1st_dim)
        Xts = slice_csr_like(indptr, data, ts_idx, exclude_1st_dim)

        # STANDARDIZE
        N = Xtr.data.shape[0]
        x_bar = Xtr.data.mean(0)
        C = ((Xtr.data.T @ Xtr.data) - N * np.outer(x_bar, x_bar)) / (N - 1.)
        P = np.linalg.inv(C)
        S_chol = np.linalg.cholesky(P).astype(Xtr.data.dtype)
        if normalization:
            mvnorm_standardize(Xtr.data, x_bar, S_chol)
            mvnorm_standardize(Xts.data, x_bar, S_chol)

        hf.close()
        return (
            {
                'train': (Xtr, ids[tr_idx]),
                'test': (Xts, ids[ts_idx]),
            },
            {'mean': x_bar, 'prec_chol': S_chol}
        )

    else:
        dset = MVVarSeqData(
            indptr = hf['indptr'][:],
            data = hf['data'][:] if not exclude_1st_dim else hf['data'][:, 1:]
        )

        # STANDARDIZE
        if normalization:
            x_bar = dset.data.mean(0)
            C = np.cov(dset.data.T)
            P = np.linalg.inv(C)
            S_chol = np.linalg.cholesky(P)
            mvnorm_standardize(dset.data, x_bar, S_chol)
        else:
            x_bar = None
            S_chol = None

        ids = [i for i in hf['ids'][:]]

        hf.close()
        return dset, ids, {'mean': x_bar, 'prec_chol': S_chol}

