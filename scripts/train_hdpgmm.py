from typing import Union
from pathlib import Path
import pickle as pkl
import argparse
import logging
import json
import sys
sys.path.append((Path(__file__).parent.parent.parent / 'pytorch-hdpgmm').as_posix())

import numpy as np
import h5py
import torch
from hdpgmm import model as hdpgmm_gpu  # gpu
from hdpgmm.data import HDFMultiVarSeqDataset


logging.basicConfig()
logger = logging.getLogger("TrainingHDPGMM")


class HDFMultiVarSeqAugDataset(HDFMultiVarSeqDataset):
    def __init__(
        self,
        h5_fn: Union[str, Path],
        whiten: bool = False,
        chunk_size: int = 1024,
        apply_aug: bool = True,
        aug_key: str = 'augmentation',
        verbose: bool = False
    ):
        """
        """
        super().__init__(h5_fn, whiten, chunk_size, verbose)
        self.apply_aug = apply_aug
        self.aug_key = aug_key

        # build original / augmentation dict
        self._data_path_dict = dict()
        self._data_path_dict[0] = 'data'

        if apply_aug:
            self._check_augmentation()
            with h5py.File(self.h5_fn, 'r') as hf:
               for i in range(len(hf[self.aug_key])):
                   self._data_path_dict[i + 1] = f'{self.aug_key}/{i:d}'

    def _check_augmentation(self):
        """
        """
        with h5py.File(self.h5_fn, 'r') as hf:
            try:
                assert self.aug_key in hf
            except Exception as e:
                print(f"Can not find key '{self.aug_key}' in the dataset!")
                raise e

    def __getitem__(
        self,
        idx: int
    ) -> tuple[int, torch.Tensor]:
        """
        """
        if self.apply_aug:
            # pick the augmented
            if np.random.rand() > 0.5:
                # augmentation
                dataset_id = np.random.randint(len(self._data_path_dict) - 1) + 1
            else:
                # no augmentation
                dataset_id = 0

            dataset_key = self._data_path_dict[dataset_id]
        else:
            dataset_key = 'data'

        with h5py.File(self.h5_fn, 'r') as hf:
            # index frames/tokens
            j0, j1 = hf['indptr'][idx], hf['indptr'][idx+1]

            # retrieve data
            x = hf[dataset_key][j0:j1]

        # whiten, if needed
        x = self.apply_whitening(x)

        # wrap to torch.Tensor
        x = torch.as_tensor(x, dtype=torch.float32)

        return (idx, x)


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
    parser.add_argument("--augmentation", default=False,
                        action=argparse.BooleanOptionalAction,
                        help="use data augmentation or not.")
    parser.add_argument("--verbose", default=True,
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

    dataset = HDFMultiVarSeqAugDataset(
        config['dataset']['path'],
        whiten = config['dataset']['whiten'],
        apply_aug = args.augmentation
    )

    ret = hdpgmm_gpu.variational_inference(
        dataset,
        warm_start_with = warm_start,
        **config['model']
    )


if __name__ == "__main__":
    main()
