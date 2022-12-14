from pathlib import Path
import argparse
import logging

import numpy as np

from ..experiment.mtat import load_mtat
from ..experiment.gtzan import load_gtzan
from ..experiment.echonest import load_echonest
from ..experiment.common import (MODEL_MAP,
                                 load_model,
                                 process_feature)


DATASET_MAP = {
    'gtzan': load_gtzan,
    'mtat': load_mtat,
    'echonest': load_echonest
}


logging.basicConfig()
logger = logging.getLogger(__name__)

np.random.seed(2022)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="extractfeat",
        description=(
            "Feature extraction from the fitted models."
        )
    )

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
    parser.add_argument("--model-path", type=str, default=None,
                        help=("path where fitted feature learner model is "
                              "located. if it the model class is given as 'G1', "
                              "model file is not required."))
    parser.add_argument("--split-path", type=str, default=None,
                        help="path where split info dataset")
    parser.add_argument("--device", type=str, default='cpu',
                        help=(
                            "specify acceleration device. "
                            "only relevant for `hdpgmm` model"
                            " {i.e., 'cpu', 'cuda:0', 'cuda:1', ...}"
                        ))
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
    TODO: now the program hang with VQCodeBook instance
          after the feature extraction is finished.
    """
    args = parse_arguments()
    if args.verbose:
        logger.setLevel(logging.INFO)

    # check if the model path
    if args.model_class != 'G1':
        assert (
            (args.model_path is not None) and
            (Path(args.model_path).exists())
        )
    else:
        assert args.model_path is None

    dataset = DATASET_MAP[args.dataset](args.dataset_path,
                                        args.split_path)

    model = load_model(args.model_path,
                       args.model_class, dataset,
                       batch_size = args.batch_size,
                       device = args.device)
    config = model.get_config()

    X, _ = process_feature(model, dataset,
                           loudness_cols=False)

    # knit output filename
    if args.model_class == 'G1':
        stem = name = args.model_class
    else:
        model_path = Path(args.model_path)
        stem = model_path.stem
        name = model_path.name

    out_fn = Path(args.out_path) / f'{args.model_class}_{args.dataset}_{stem}.npz'

    # check and make parent directory if necessary
    out_fn.parent.mkdir(exist_ok=True, parents=True)

    # save them
    np.savez(out_fn, feature=X, ids=dataset.data.ids,
             dataset=np.array(args.dataset),
             model_class=np.array(config['model_class']),
             model_filename=np.array(name))


if __name__ == "__main__":
    raise SystemExit(main())
