from typing import Optional
import argparse
from pathlib import Path
import pickle as pkl

import numpy as np
from scipy import sparse as sp

import hdpgmm
from hdpgmm.data import HDFMultiVarSeqDataset


def load_lastfm(
    tid_list_fn: str,
    tag_list_fn: str,
    triplet_fn: str,
    const_value: Optional[float] = None
) -> dict[str, object]:
    """
    """
    # load tids
    with Path(tid_list_fn).open('r') as fp:
        tids = []
        tids2id = {}
        for line in fp:
            tid = line.replace('\n', '')
            tids2id[tid] = len(tids)
            tids.append(tid)

    # load tags
    with Path(tag_list_fn).open('r') as fp:
        tags = []
        tags2id = {}
        for line in fp:
            tag = line.replace('\n', '')
            tags2id[tag] = len(tags)
            tags.append(tag)

    # load triplets
    with Path(triplet_fn).open('r') as fp:
        rows, cols, data = [], [], []
        for line in fp:
            i, j, v = line.replace('\n', '').split(',')
            i, j, v = int(i), int(j), float(v)
            rows.append(i)
            cols.append(j)
            data.append(v)

        if const_value is not None:
            data = np.full((len(rows),), const_value)

        X = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(len(tids), len(tags))
        ).tocsr()

    return {
        'mat': X,
        'tids': tids,
        'tids2id': tids2id,
        'tags': tags,
        'tags2id': tags2id
    }


def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract_topic_terms",
        description=(
            "Extracts topic-terms using MSD-Lastfm tag mapping"
        )
    )

    parser.add_argument("model_fn", type=str,
                        help="filename of fitted feature learner model (hdpgmm) is located")
    parser.add_argument("data_fn", type=str,
                        help="filename of main training dataset (hdf)")
    parser.add_argument("lastfm_tids_fn", type=str,
                        help="track-id list of lastfm-msd tag database")
    parser.add_argument("lastfm_tags_fn", type=str,
                        help="tag list of lastfm-msd tag database")
    parser.add_argument("lastfm_triplets_fn", type=str,
                        help="trackid-tag-count triplet list of lastfm-msd tag database")
    parser.add_argument("out_path", type=str,
                        help="output path")
    parser.add_argument('--topk-corpus', type=int, default=20,
                        help='the number of top-K components to be printed on corpus level')
    parser.add_argument('--topk-component', type=int, default=5,
                        help='the number of top-K tags within the component')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='the number of samples in a mini-batch')
    parser.add_argument('--device', type=str, default='cpu',
                        choices={'cpu', 'cuda'},
                        help='accelerator')
    parser.add_argument('--verbose', default=True,
                        action=argparse.BooleanOptionalAction,
                        help="set verbosity")
    return parser.parse_args()



def main():
    """
    """
    model_fn = Path(args.model_fn)
    out_fn = Path(args.out_path) / f'component_tags_{model_fn.stem}.json'

    # load model
    with model_fn.open('rb') as fp:
        model = pkl.load(fp)

    # load data
    dataset = HDFMultiVarSeqDataset(args.data_fn)
    if model.whiten_params is not None:
        dataset._whitening_params = model.whiten_params
        dataset.whiten = True
    else:
        dataset._whitening_params = None
        dataset.whiten = False

    # load lastfm data
    lastfm = load_lastfm(
        args.lastfm_tids_fn,
        args.lastfm_tags_fn,
        args.lastfm_triplets_fn,
        const_value = 1.
    )

    # extract feature
    features = hdpgmm.model.infer_documents(
        dataset, model,
        n_max_inner_iter = 1000,
        e_step_tol = 1e-6,
        max_len = 2600,
        batch_size = args.batch_size,
        device = args.device
    )
    pi = features['responsibility'].detach().cpu().numpy()
    ids = dataset._hf['ids'][:].astype('U')

    mat = lastfm['mat'][[lastfm['tids2id'][j] for j in ids]]
    idf = (
        np.log(lastfm['mat'].shape[0])
        - np.array(np.log(lastfm['mat'].sum(0) + 1))[0]
    )

    # compute loadings
    f = pi.T @ mat

    top_comps = np.argsort(-pi.sum(0))[:args.topk_corpus]
    comp_tags = []
    for k in top_comps.tolist():
        s = f[k] * idf
        idx = np.argpartition(-s, kth=args.topk_component)[:args.topk_component]
        top_tags = [lastfm['tags'][t] for t in idx[np.argsort(-s[idx])]]
        comp_tags.append((k, top_tags))

    # save
    with out_fn.open('w') as fp:
        json.dump(comp_tags, fp)


if __name__ == "__main__":
    main()
