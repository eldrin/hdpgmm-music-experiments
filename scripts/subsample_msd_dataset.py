from pathlib import Path
import argparse
import logging

import numpy as np
import h5py
from tqdm import tqdm


logging.basicConfig()
logger = logging.getLogger("MSDDataSubsampling")

H5PY_STR = h5py.special_dtype(vlen=str)

np.random.seed(2022)


def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subsample_msd",
        description=(
            "this program sub-samples the MSD feature dataset (in hdf) "
            "into a number of subsets with a certain subset size."
        )
    )

    parser.add_argument("hdf_fn", type=str,
                        help="source HDF dataset filename")
    parser.add_argument("out_path", type=str,
                        help="root directory where the outputs are stored")
    parser.add_argument("--num-subsets", type=int, default=5,
                        help="the number of subsampled dataset to be made")
    parser.add_argument("--subsample", type=int, default=20000,
                        help="the number of subsample that is going to be processed")
    parser.add_argument('--verbose', default=True,
                        action=argparse.BooleanOptionalAction,
                        help="set verbosity")
    return parser.parse_args()


def main() -> None:
    """
    """
    args = parse_arguments()

    for i in range(args.num_subsets):
        with h5py.File(args.hdf_fn) as hf:

            # get subset indices
            rnd_idx = np.random.choice(hf['ids'].shape[0],
                                       args.subsample,
                                       False)

            # knit filename
            out_fn = (
                Path(args.out_path) /
                    (Path(args.hdf_fn).stem
                     + f'_subset{args.subsample:d}_{i+1:d}.h5')
            )

            with h5py.File(out_fn, 'w') as hfo:
                total_len = np.ediff1d(hf['indptr'][:])[rnd_idx].sum()
                feature_dim = hf['data'].shape[-1]
                dtype = hf['data'].dtype
                dataset = hfo.create_dataset('data',
                                             (total_len, feature_dim),
                                             dtype=dtype)
                msd_ids = []
                indptr = [0]
                last = indptr[-1]
                with tqdm(total=len(rnd_idx), ncols=80, disable=not args.verbose) as prog:
                    for j in rnd_idx:
                        j0, j1 = hf['indptr'][j], hf['indptr'][j+1]
                        next_last = last + j1 - j0
                        x = hf['data'][j0:j1]
                        dataset[last:next_last] = x
                        indptr.append(next_last)
                        msd_ids.append(hf['ids'][j])
                        last = next_last
                        prog.update()

                hfo.create_dataset('indptr', data=np.array(indptr), dtype='i4')
                hfo.create_dataset('ids', data=np.array(msd_ids, dtype=H5PY_STR))


if __name__ == "__main__":
    main()
