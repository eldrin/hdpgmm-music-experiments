from pathlib import Path
import json
import argparse
import logging
import sys

sys.path.append(Path(__file__).parent.parent.as_posix())

import numpy as np

from src.experiment.common import MODEL_MAP
from src.experiment.gtzan import run_experiment

logging.basicConfig()
logger = logging.getLogger("TestGTZAN")

np.random.seed(2022)


def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="test_gtzan",
        description=(
            "Testing fitted representation for GTZAN genre classification."
        )
    )

    parser.add_argument("model_path", type=str,
                        help="path where fitted feature learner model is located")
    parser.add_argument("model_class", type=str,
                        choices=set(MODEL_MAP.keys()),
                        help="class of the feature learner model")
    parser.add_argument("dataset_path", type=str,
                        help="path where pre-processed dataset (hdf5) is located")
    parser.add_argument("split_path", type=str,
                        help="path where split info dataset")
    parser.add_argument("out_path", type=str,
                        help="root directory where fitted model is stored")
    parser.add_argument('--exclude-1st-dim', default=False,
                        action=argparse.BooleanOptionalAction,
                        help="exclude loudness dim of MFCC")
    parser.add_argument('--exclude-chroma-dims', default=False,
                        action=argparse.BooleanOptionalAction,
                        help="exclude chroma features")
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

    # load data
    logger.info('Start Testing!')
    results = run_experiment(args.model_class,
                             args.model_path,
                             args.dataset_path,
                             args.split_path,
                             args.exclude_1st_dim,
                             args.exclude_chroma_dims,
                             n_jobs=args.n_jobs)
    logger.info('Testing Done!')

    logger.info('Saving...')
    task = results['task']
    filename = Path(args.model_path).name
    model_class = results['model_class']
    out_fn = Path(args.out_path) / f'{model_class}_{task}.json'
    with out_fn.open('w') as fp:
        json.dump(results, fp)
    logger.info('Saving done!')


if __name__ == "__main__":
    main()
