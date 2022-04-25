from pathlib import Path
import pickle as pkl
import argparse
import logging
import json
import sys
sys.path.append((Path(__file__).parent.parent.parent / 'pytorch-hdpgmm').as_posix())

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
    parser.add_argument("--warm-start", default=None,
                        help=("filename of the checkpoint of the model for "
                              "the warm-start"))
    parser.add_argument("--device", default=None,
                        help=("the main device that 'pytorch' uses to compute."
                              "it overrides the configuration setup if given"))
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
        if args.device:
            config['model']['device'] = args.device

    if args.warm_start is not None:
        if Path(args.warm_start).exists():
            with Path(args.warm_start).open('rb') as fp:
                warm_start = pkl.load(fp)
        else:
            raise FileNotFoundError(
                "[ERROR] can't find the given model check point file!"
            )
    else:
        warm_start = None

    dataset = HDFMultiVarSeqDataset(
        config['dataset']['path'],
        whiten = config['dataset']['whiten']
    )

    ret = hdpgmm_gpu.variational_inference(
        dataset,
        warm_start_with = warm_start,
        **config['model']
    )


if __name__ == "__main__":
    main()
