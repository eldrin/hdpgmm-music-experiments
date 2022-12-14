from typing import Optional
from pathlib import Path
import argparse
import logging

import numpy as np

from .experiment.mtat import load_mtat
from .experiment.gtzan import load_gtzan
from .experiment.echonest import load_echonest
from .experiment.common import (MODEL_MAP,
                                load_model,
                                process_feature)


DATASET_MAP = {
    'gtzan': load_gtzan,
    'mtat': load_mtat,
    'echonest': load_echonest
}
DEFAULT_SEED = 2022


logging.basicConfig()
logger = logging.getLogger(__name__)


def knit_path(
    msd_id: str,
    out_path: Path
) -> Path:
    """ knit full path using target directory and MSD id
    """
    # build out dir
    out_dir = out_path / msd_id[2] / msd_id[3] / msd_id[4]

    # make folder if not exsists
    out_dir.mkdir(parents=True, exist_ok=True)

    # knit the output filename
    p = out_dir / f'{msd_id}.npy'

    return p


def extract_feature(
    model_class: str,
    dataset_type: str,
    dataset_path: str,
    out_path: str,
    split_path: str,
    batch_size: int,
    device: str,
    model_path: Optional[str] = None,
) -> None:
    """ extract feature from feature learners

    Args:
        model_class: type of model to be processed.
                     {`hdpgmm`, `vqcodebook`, `G1`, `precomputed`}
        dataset_type: type of dataset to be processed.
                      { `gtzan`, `mtat`, `echonest`}
        dataset_path: path of the dataset.
        out_path: output path where the result is stored.
        split_path: path of the data split file.
        batch_size: size of the mini-batch if applied.
        device: main computing device. {`cpu`, `cuda`, `cuda:n`, ...}
        model_path: path of the pre-trained model (optional)
    """
    # check if the model path
    if model_class != 'G1':
        assert (
            (model_path is not None) and
            (Path(model_path).exists())
        )
    else:
        assert model_path is None

    dataset = DATASET_MAP[dataset_type](dataset_path,
                                        split_path)

    model = load_model(model_path,
                       model_class,
                       dataset,
                       batch_size = batch_size,
                       device = device)
    config = model.get_config()

    X, _ = process_feature(model, dataset,
                           loudness_cols=False)

    # knit output filename
    if model_class == 'G1':
        stem = name = model_class
    else:
        model_path = Path(model_path)
        stem = model_path.stem
        name = model_path.name

    out_fn = Path(out_path) / f'{model_class}_{dataset}_{stem}.npz'

    # check and make parent directory if necessary
    out_fn.parent.mkdir(exist_ok=True, parents=True)

    # save them
    np.savez(out_fn, feature=X, ids=dataset.data.ids,
             dataset=np.array(dataset),
             model_class=np.array(config['model_class']),
             model_filename=np.array(name))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="utils",
        description=(
            "Feature extraction from the fitted models."
        )
    )

    subparsers = parser.add_subparsers(title="command",
                                       dest="command",
                                       help="sub-command help")
    subparsers.required = True

    base_subparser = argparse.ArgumentParser(add_help=False)
    base_subparser.add_argument("-p", "--path", type=str, default="./",
                                help="path where the output stored")
    base_subparser.add_argument("-s", "--random-seed", type=int, default=DEFAULT_SEED,
                                help="random seed to fix the random generator")
    base_subparser.add_argument('--verbose', default=True,
                                action=argparse.BooleanOptionalAction,
                                help="set verbosity")

    # `extract` sub command ===================================================
    extract = subparsers.add_parser(
        'extract', parents=[base_subparser],
        help='compute learned feature from subset of feature models.'
    )

    extract.add_argument("model_class", type=str,
                        choices=set(MODEL_MAP.keys()),
                        help="class of the feature learner model")
    extract.add_argument("dataset_path", type=str,
                        help="path where pre-processed dataset (hdf5) is located")
    extract.add_argument("dataset", type=str,
                        choices=set(DATASET_MAP.keys()),
                        help="dataset (task) name to be computed")
    extract.add_argument("--model-path", type=str, default=None,
                        help=("path where fitted feature learner model is "
                              "located. if it the model class is given as 'G1', "
                              "model file is not required."))
    extract.add_argument("--split-path", type=str, default=None,
                        help="path where split info dataset")
    extract.add_argument("--device", type=str, default='cpu',
                        help=(
                            "specify acceleration device. "
                            "only relevant for `hdpgmm` model"
                            " {i.e., 'cpu', 'cuda:0', 'cuda:1', ...}"
                        ))
    extract.add_argument('-m', '--batch-size', type=int, default=1024,
                        help='number of samples per minibatch for feature extraction')
    extract.add_argument('-j', '--n-jobs', type=int, default=2,
                        help='number of cores for extract HDPGMM features')

    return parser.parse_args()


def main():
    """
    TODO: now the program hang with VQCodeBook instance
          after the feature extraction is finished.
    """
    args = parse_arguments()
    if args.verbose:
        logger.setLevel(logging.INFO)

    # set random seed
    np.random.seed(args.random_seed)

    if args.command == "extract":

        extract_feature(
            args.model_class,
            args.dataset,
            args.dataset_path,
            args.out_path,
            args.split_path,
            args.batch_size,
            args.device,
            args.model_path,
        )

    else:
        ValueError('[ERROR] only `extract` subcommand is available!')


if __name__ == "__main__":
    raise SystemExit(main())
