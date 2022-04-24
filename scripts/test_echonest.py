from pathlib import Path
import json
import argparse
import logging
import sys

sys.path.append(Path(__file__).parent.parent.as_posix())

import numpy as np

from src.experiment.common import MODEL_MAP, SIM_FUNC_MAP
from src.experiment.echonest import run_experiment

logging.basicConfig()
logger = logging.getLogger("TestEchonest")

np.random.seed(2022)


def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="test_echonest",
        description=(
            "Testing fitted representation for Echonest / MSD music recomendation."
        )
    )

    parser.add_argument("model_path", type=str,
                        help="path where fitted feature learner model is located")
    parser.add_argument("model_class", type=str,
                        choices=set(MODEL_MAP.keys()),
                        help="class of the feature learner model")
    parser.add_argument("dataset_path", type=str,
                        help="path where pre-processed dataset (hdf5) is located")
    parser.add_argument("out_path", type=str,
                        help="root directory where fitted model is stored")
    parser.add_argument('-m', '--batch-size', type=int, default=1024,
                        help='number of samples per minibatch for feature extraction')
    parser.add_argument('-j', '--n-jobs', type=int, default=2,
                        help='number of cores for extract HDPGMM features')
    parser.add_argument('-s', '--similarity', type=str, default='cosine',
                        choices=set(SIM_FUNC_MAP.keys()),
                        help="set which similarity is used for KNN")
    parser.add_argument('--cuda', default=False,
                        action=argparse.BooleanOptionalAction,
                        help="user CUDA acceleration")
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
                             batch_size=args.batch_size,
                             valid_user_ratio = 0.002,
                             test_user_ratio = 0.002,
                             similarity=args.similarity,
                             device='cuda' if args.cuda else 'cpu',
                             n_jobs=args.n_jobs)
    logger.info('Testing Done!')

    logger.info('Saving...')
    task = results['task']
    filename = Path(args.model_path).name
    model_class = results['model_class']
    out_fn = Path(args.out_path) / f'{model_class}_{task}_{filename}.json'
    with out_fn.open('w') as fp:
        json.dump(results, fp)
    logger.info('Saving done!')


if __name__ == "__main__":
    main()
