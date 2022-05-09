from pathlib import Path
import json
import argparse
import logging
import sys

sys.path.append(Path(__file__).parent.parent.as_posix())

import numpy as np

from src.models import HDPGMM
from src.experiment.mtat import load_mtat
from src.experiment.gtzan import load_gtzan
from src.experiment.echonest import load_echonest
from src.experiment.common import (MODEL_MAP,
                                   load_model,
                                   process_feature)


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

    parser.add_argument("model_path", type=str,
                        help="path where fitted feature learner model is located")
    parser.add_argument("model_class", type=str,
                        choices=set(MODEL_MAP.keys()),
                        help="class of the feature learner model")
    parser.add_argument("dataset_path", type=str,
                        help="path where pre-processed dataset (hdf5) is located")
    parser.add_argument("dataset", type=str,
                        choices=set(DATASET_MAP.keys()),
                        help="dataset (task) name to be computed")
    parser.add_argument("out_path", type=str,
                        help="root directory where fitted model is stored")
    parser.add_argument("--split-path", type=str, default=None,
                        help="path where split info dataset")
    parser.add_argument('-m', '--batch-size', type=int, default=1024,
                        help='number of samples per minibatch for feature extraction')
    parser.add_argument('-j', '--n-jobs', type=int, default=2,
                        help='number of cores for extract HDPGMM features')
    parser.add_argument('--verbose', default=True,
                        action=argparse.BooleanOptionalAction,
                        help="set verbosity")
    return parser.parse_args()


def main():
    """
    """
    args = parse_arguments()
    if args.verbose:
        logger.setLevel(logging.INFO)

    dataset = DATASET_MAP[args.dataset](args.dataset_path,
                                        args.split_path)
    model = load_model(args.model_path, args.model_class, dataset,
                       batch_size = args.batch_size)
    config = model.get_config()

    X, y = process_feature(model, dataset,
                           loudness_cols = False)

    # save them
    fn = Path(args.model_path)
    out_fn = Path(args.out_path) / f'{args.model_class}_{args.dataset}_{fn.stem}.npz'
    out_fn.parent.mkdir(exist_ok=True, parents=True)
    np.savez(out_fn, feature=X, ids=dataset.data._hf['ids'][:],
             dataset=np.array(args.dataset),
             model_class=np.array(config['model_class']),
             model_filename=np.array(fn.name))


if __name__ == "__main__":
    main()
