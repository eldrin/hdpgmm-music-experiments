import argparse
import logging

import numpy as np

from ..config import DEFAULTS
from ..preprocessing import mtat, gtzan, echonest, msd

logging.basicConfig()
logger = logging.getLogger(__name__)

DEFAULT_SEED = 2022


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="preproc",
        description=(
            "pre-processing downstream datasets. "
            "This script process the preview songs into "
            "per-song mel spectrogram, and save them into numpy files."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    subparsers = parser.add_subparsers(title="command",
                                       dest='command',
                                       help='sub-command help')
    subparsers.required = True

    base_subparser = argparse.ArgumentParser(add_help=False)
    base_subparser.add_argument("-p", "--path", type=str, default="./",
                                help="path where the output stored")
    base_subparser.add_argument("-s", "--random-seed", type=int,
                                default=DEFAULT_SEED,
                                help="random seed to fix the random generator")
    base_subparser.add_argument("--out-name", type=str, default='mtat_feature',
                                help='filename stem for the output dataset file')
    base_subparser.add_argument("--n-fft", type=float, default=DEFAULTS['n_fft'],
                                help=("the size of each audio frame, which is to "
                                      "be FFT-ed"))
    base_subparser.add_argument("--hop-sz", type=float,
                                default=DEFAULTS['hop_sz'],
                                help="the amount to slide through")
    base_subparser.add_argument("-j", "--n-jobs", type=int, default=1,
                                help=("the number of processed used for the "
                                      "multiprocessing."))
    base_subparser.add_argument('--verbose', default=True,
                                action=argparse.BooleanOptionalAction,
                                help="set verbosity")

    # `mtat` sub command ======================================================
    mtat = subparsers.add_parser(
        'mtat', parents=[base_subparser],
        help="processing MTAT dataset."
    )
    mtat.add_argument("mtat_path", type=str,
                        help="""path where MagnaTagATune dataset is unzipped.
we assume the folder contains both music audio files
and the annotation file (annotations_final.csv).
the structure the script is expecting is as follows:\n
  -- mtat_path/
   |-- audio/
   | |-- 0/
   | |-- 1/
   | |.../
   | |-- f/
   |-- annotations_final.csv

""")

    # `gtzan` sub command =====================================================
    gtzan = subparsers.add_parser(
        'gtzan', parents=[base_subparser],
        help="processing GTZAN dataset"
    )
    gtzan.add_argument("gtzan_path", type=str,
                       help="path where GTZAN dataset is unzipped")

    # `echonest` subcommand ===================================================
    echonest = subparsers.add_parser(
        'echonest', parents=[base_subparser],
        help="processing MSD-Echonest dataset"
    )
    echonest.add_argument("msd_mp3_path", type=str,
                          help="MSD mp3 file root path")
    echonest.add_argument("msd_path_info", type=str,
                          help=("file contains the map from MSD id to the "
                                "subpath of the audio (.pkl)"))
    echonest.add_argument("msd_song2track", type=str,
                          help=("file contains the map from MSD song id to "
                                "the track id (.pkl)"))
    echonest.add_argument("echonest_triplet", type=str,
                          help=("filename of Echonest/MSD interaction "
                                "triplet data (.txt)"))

    # `msd` subcommand =======================================================
    msd = subparsers.add_parser(
        'msd', parents=[base_subparser],
        help="processing MSD dataset (subset)"
    )
    msd.add_argument("msd_mp3_path", type=str,
                     help="MSD mp3 file root path")
    msd.add_argument("msd_path_info", type=str,
                     help=("file contains the map from MSD id to the "
                           "subpath of the audio (.pkl)"))
    msd.add_argument("--target-list", type=str, default=None,
                     help="text file contains target MSD id per line.")
    msd.add_argument("--subsample", type=int, default=200_000,
                     help=("the number of samples to be sub-sampled "
                           "for dataset building"))

    return parser.parse_args()


def main():
    """
    """
    args = parse_arguments()
    if args.verbose:
        logger.setLevel(logging.INFO)

    # set random seed
    np.random.seed(args.random_seed)

    if args.command == 'mtat':
        mtat.process(
            args.mtat_path, args.path, args.out_name, args.n_fft,
            args.hop_sz, args.n_jobs, args.verbose
        )

    elif args.command == 'gtzan':
        gtzan.process(
            args.gtzan_path, args.path, args.out_name, args.n_fft,
            args.hop_sz, args.n_jobs, args.verbose
        )

    elif args.command == 'echonest':
        echonest.process(
            args.msd_mp3_path, args.msd_path_info, args.song2track,
            args.echonest_triplet, args.path, args.out_name,
            args.n_fft, args.hop_sz, args.n_jobs, args.verbose
        )

    elif args.command == 'msd':
        msd.process(
            args.msd_mp3_path, args.msd_path_info, args.path, args.out_name,
            args.target_list, args.n_fft, args.hop_sz, args.subsample,
            args.n_jobs, args.verbose
        )

    else:
        ValueError(
            '[ERROR] only `msd`, `mtat`, `gtzan`, and '
            '`echoenst` subcommand is available!'
        )


if __name__ == "__main__":
    raise SystemExit(main())
