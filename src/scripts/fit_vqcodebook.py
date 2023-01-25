from pathlib import Path
import argparse
import logging

import numpy as np

from hdpgmm.data import HDFMultiVarSeqDataset

from ..models import VQCodebook

logging.basicConfig()
logger = logging.getLogger(__name__)

np.random.seed(2022)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fit_vqcodebook",
        description=(
            "Fitting Vector-Quantization (VC) Codebooks on the frame-level of features"
            ", which will be used one of the baselines."
        )
    )

    parser.add_argument("dataset_path", type=str,
                        help="path where pre-processed melspec dataset (hdf5) is located")
    parser.add_argument("out_path", type=str,
                        help="root directory where fitted model is stored")
    parser.add_argument("out_fn_prefix", type=str, default="vqcodebook",
                        help="prefix for the output file name")
    parser.add_argument("-k", "--n-components", type=int, default=128,
                        help="the number of components.")
    parser.add_argument('--whiten', default=False,
                        action=argparse.BooleanOptionalAction,
                        help="whitening the input feature")
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
    logger.info('Loading Data!...')
    dataset = HDFMultiVarSeqDataset(
        args.dataset_path,
        whiten = args.whiten
    )
    logger.info('Data Loaded!...')

    logger.info('Start Inference!')
    vq = VQCodebook(args.n_components)
    vq.fit(dataset)
    logger.info('Inference Done!')

    logger.info('Saving...')
    fn = f'{args.out_fn_prefix}_k{args.n_components:d}.joblib'
    out_fn = Path(args.out_path) / fn
    vq.save(out_fn.as_posix())
    logger.info('Saving done!')


if __name__ == "__main__":
    raise SystemExit(main())
