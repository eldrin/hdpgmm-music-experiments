from pathlib import Path
import json
import argparse
import logging
import sys

sys.path.append(Path(__file__).parent.parent.as_posix())

import numpy as np
from tqdm import tqdm

from src.experiment.mtat import load_mtat
from src.experiment.gtzan import load_gtzan
from src.experiment.echonest import load_echonest


DATASET_MAP = {
    'gtzan': load_gtzan,
    'mtat': load_mtat,
    'echonest': load_echonest
}


logging.basicConfig()
logger = logging.getLogger("ExtractFeature")

np.random.seed(2022)


def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extractfeat",
        description=(
            "Feature extraction from the fitted models."
        )
    )

    parser.add_argument("dataset_path", type=str,
                        help="path where pre-processed dataset (hdf5) is located")
    parser.add_argument("dataset", type=str,
                        choices=set(DATASET_MAP.keys()),
                        help="dataset (task) name to be computed")
    parser.add_argument("out_path", type=str,
                        help="root directory where fitted model is stored")
    parser.add_argument("--split-path", type=str, default=None,
                        help="path where split info dataset")
    parser.add_argument('--verbose', default=True,
                        action=argparse.BooleanOptionalAction,
                        help="set verbosity")
    return parser.parse_args()


def main():
    """
    TODO: now the program hang with VQCodeBook instance
          after the feature extraction is finished.
    """
    args = parse_arguments()
    if args.verbose:
        logger.setLevel(logging.INFO)

    dataset = DATASET_MAP[args.dataset](args.dataset_path,
                                        args.split_path)

    # extracting G1 feature
    X = np.empty((len(dataset.data), dataset.data._hf['data'].shape[1] * 2),
                 dtype=dataset.data._hf['data'].dtype)
    with tqdm(total=len(dataset.data), ncols=80, disable=not args.verbose) as prog:
        for j in range(len(dataset.data)):
            x = dataset.data[j][1].detach().cpu().numpy()
            X[j] = np.r_[x.mean(0), x.std(0)]
            prog.update()

    # save them
    out_fn = Path(args.out_path) / f'g1_{args.dataset}.npz'
    out_fn.parent.mkdir(exist_ok=True, parents=True)
    np.savez(out_fn, feature=X, ids=dataset.data._hf['ids'][:],
             dataset=np.array(args.dataset),
             model_class=np.array('g1'),
             model_filename=np.array('null'))
    dataset.data._hf.close()  # is it really necessary?


if __name__ == "__main__":
    main()
