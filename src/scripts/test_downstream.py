from typing import Optional
from pathlib import Path
import json
import argparse
import logging
from logging import Logger
import sys

import numpy as np

from ..experiment.common import MODEL_MAP, SIM_FUNC_MAP
from ..experiment.echonest import run_experiment as run_echonest
from ..experiment.gtzan import run_experiment as run_gtzan
from ..experiment.mtat import run_experiment as run_mtat


logging.basicConfig()
logger = logging.getLogger(__name__)


TASKS = {'gtzan', 'echonest', 'mtat'}


DEFAULT_SEED = 2022


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="test_downstream",
        description=(
            "Test downstream task using pre-trained representation."
        )
    )

    subparsers = parser.add_subparsers(title="command",
                                       dest="command",
                                       help="sub-command help")
    subparsers.required = True

    base_subparser = argparse.ArgumentParser(add_help=False)
    base_subparser.add_argument("model_path", type=str,
                                help="path where fitted feature learner model is located")
    base_subparser.add_argument("model_class", type=str,
                                choices=set(MODEL_MAP.keys()),
                                help="class of the feature learner model")
    base_subparser.add_argument("dataset_path", type=str,
                                help="path where pre-processed dataset (hdf5) is located")
    base_subparser.add_argument("-p", "--path", type=str, default="./",
                                help="path where the output stored")
    base_subparser.add_argument('-m', '--batch-size', type=int, default=1024,
                                help='number of samples per minibatch for feature extraction')
    base_subparser.add_argument('-j', '--n-jobs', type=int, default=2,
                                help='number of cores for extract HDPGMM features')
    base_subparser.add_argument("--random-seed", type=int, default=DEFAULT_SEED,
                                help="random seed to fix the random generator")
    base_subparser.add_argument('--verbose', default=True,
                                action=argparse.BooleanOptionalAction,
                                help="set verbosity")

    # `echonest` sub command ===================================================
    echonest = subparsers.add_parser(
        'echonest', parents=[base_subparser],
        help=(
            'testing fitted representation for Echonest / MSD music '
            'recommendation'
        )
    )
    echonest.add_argument('-s', '--similarity', type=str, default='cosine',
                          choices=set(SIM_FUNC_MAP.keys()),
                          help="set which similarity is used for KNN")
    echonest.add_argument('--accelerator', type=str, default='cpu',
                          help="set the acceleration device")

    # `gtzan` sub command ===================================================
    gtzan = subparsers.add_parser(
        'gtzan', parents=[base_subparser],
        help=(
            'testing fitted representation for GTZAN music genre '
            'classification'
        )
    )
    gtzan.add_argument("split_path", type=str,
                       help="path where split info dataset")
    _ = gtzan

    # `gtzan` sub command ===================================================
    mtat = subparsers.add_parser(
        'mtat', parents=[base_subparser],
        help=(
            'testing fitted representation for MTAT music auto '
            'tagging.'
        )
    )
    mtat.add_argument("split_path", type=str,
                      help="path where split info dataset")
    mtat.add_argument('--accelerator', type=str, default='cpu',
                      help="set the acceleration device")

    return parser.parse_args()



def main():
    """
    """
    args = parse_arguments()
    if args.verbose:
        logger.setLevel(logging.INFO)

    # set random seed
    np.random.seed(args.random_seed)

    logger.info('Start Testing!')

    if args.command == "echonest":
        results = run_echonest(
            model_class = args.model_class,
            model_fn = args.model_path,
            lfm1k_h5_fn = args.dataset_path,
            batch_size = args.batch_size,
            n_jobs = args.n_jobs,
            similarity = args.similarity,
            device=args.accelerator
        )

    elif args.command == "gtzan":
        results = run_gtzan(
            model_class = args.model_class,
            model_fn = args.model_path,
            gtzan_fn = args.dataset_path,
            gtzan_split_fn = args.split_path,
            n_jobs= args.n_jobs
        )

    elif args.command == "mtat":
        results = run_mtat(
            args.model_class,
            args.model_path,
            args.dataset_path,
            args.split_path,
            batch_size=args.batch_size,
            n_jobs=args.n_jobs,
            accelerator=args.accelerator
        )

    else:
        raise ValueError(
            "[ERROR] only `echonest`, `gtzan`, and `mtat` are supported!"
        )

    logger.info('Testing Done!')

    logger.info('Saving...')
    task = results['task']
    filename = Path(args.model_path).name
    model_class = results['model_class']
    out_fn = Path(args.path) / f'{model_class}_{task}_{filename}.json'
    with out_fn.open('w') as fp:
        json.dump(results, fp)
    logger.info('Saving done!')


if __name__ == "__main__":
    raise SystemExit(main())
