from pathlib import Path
import argparse
import logging
import json
import sys
sys.path.append(Path(__file__).parent.parent.as_posix())

from hdpgmm import model as hdpgmm_gpu  # gpu
from hdpgmm.data import HDFMultiVarSeqDataset


logging.basicConfig()
logger = logging.getLogger("TrainingHDPGMM")


def parse_arguments():
    """
    """
    parser = argparse.ArgumentParser(
        prog="train_hdpgmm",
        description=""
    )
    parser.add_argument("config", type=str,
                        help="filename for the training configuration")
    parser.add_argument('--verbose', default=True,
                        action=argparse.BooleanOptionalAction,
                        help="set verbosity")
    args = parser.parse_args()
    return args


def main():
    """
    """
    args = parse_arguments()
    if args.verbose:
        logger.setLevel(logging.INFO)

    with Path(args.config).open('r') as fp:
        config = json.load(fp)

    dataset = HDFMultiVarSeqDataset(
        config['dataset']['path'],
        whiten=config['dataset']['whiten']
    )

    ret = hdpgmm_gpu.variational_inference(
        dataset,
        **config['model']
    )


if __name__ == "__main__":
    main()
