from typing import Callable

import numpy as np
from scipy import sparse as sp

import torch
import torch.nn.functional as F

from tqdm import tqdm


def js_div(
    P: torch.Tensor,
    Q: torch.Tensor
) -> torch.Tensor:
    """
    """
    assert P.dim() == 2 and Q.dim() == 2
    assert P.shape[-1] == Q.shape[-1]

    M = (P[:, None] + Q[None, :]) / 2.
    dp = F.kl_div(M.log(), P[:, None], reduction='none', log_target=False).sum(-1)
    dq = F.kl_div(M.log(), Q[None, :], reduction='none', log_target=False).sum(-1)
    return ((dp + dq) / 2.)**.5


def cos_dist(
    P: torch.Tensor,
    Q: torch.Tensor
) -> torch.Tensor:
    """
    """
    assert P.dim() == 2 and Q.dim() == 2
    assert P.shape[-1] == Q.shape[-1]

    return 1. - F.cosine_similarity(P[:, None], Q[None], dim=2)


def build_knn_sim_csr(
    X: torch.Tensor,
    k: int,
    sim_func: Callable = lambda p, q: 1. - cos_dist(p, q),
    chunk_size: int = 32,
    verbose: bool = False
) -> sp.csr_matrix:  # sparse_csr
    """
    compute and output top-k similarity matrix M

    reference:
        [1]M. Deshpande and G. Karypis, “Item-based top- N recommendation algorithms,” ACM Trans. Inf. Syst., vol. 22, no. 1, pp. 143–177, Jan. 2004, doi: 10.1145/963770.963776.
    """
    M = X.shape[0]
    indptr = [0]
    indices = []
    values = []
    n_chunks = M // chunk_size + (M % chunk_size != 0)
    with tqdm(total=n_chunks, ncols=80, disable=not verbose) as prog:
        for i in range(n_chunks):
            batch_ix = torch.arange(
                i * chunk_size,
                min((i+1) * chunk_size, M)
            )
            P = X[batch_ix]

            S = sim_func(P, X)
            S[torch.arange(P.shape[0]), batch_ix] = 0.
            y = torch.topk(S, k=k, dim=1)

            for j in range(P.shape[0]):
                indptr.append(indptr[-1] + k)
            indices.append(y.indices.ravel())
            values.append(y.values.ravel())
            prog.update()

    # TODO: matmul for 2 sparse matrices are not yet supported from torch
    #       it'll probably added in next release
    # indptr = torch.as_tensor(indptr).to(X.device)
    # indices = torch.concat(indices)
    # values = torch.concat(values)
    # csr = torch.sparse_csr_tensor(indptr, indices, values, dtype=X.dtype)

    indptr = np.array(indptr)
    indices = torch.concat(indices).detach().cpu().numpy()
    values = torch.concat(values).detach().cpu().numpy()
    csr = sp.csr_matrix((values, indices, indptr), shape=(M, M))

    return csr


# def _predict_batch(
#     interaction_batch: sp.csr_matrix,
#     k_sim_mat: sp.csr_matrix,
#     topk: int = 500
# ) -> npt.ArrayLike:
#     """
#     """
#     Y = (interaction_batch @ k_sim_mat).asarray()

