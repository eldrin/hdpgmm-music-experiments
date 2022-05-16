from pathlib import Path
import copy
from functools import partial
import argparse
import json


############
# Argparser
############


def range_limited_float_type(arg, min_val, max_val):
    """ Type function for argparse - a float within some predefined bounds """
    try:
        f = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < min_val or f > max_val:
        raise argparse.ArgumentTypeError("Argument must be < " + str(max_val) + "and > " + str(min_val))
    return f


parser = argparse.ArgumentParser(
    prog="gen_exp_config",
    description=""
)

parser.add_argument("config_out", type=str,
                    help="path to output the generated configuration files")
parser.add_argument("train_data", type=str,
                    help=("pre-processed HDF (.h5) file contains training data "
                          "(the features and indices)"))
parser.add_argument("model_out", type=str,
                    help=("path that's inserted to the foncifg, where the "
                          "trained model files outputed"))
parser.add_argument("-k", "--max-components-corpus", type=int, default=256,
                    help="the maximum number of the corpus-level components")
parser.add_argument("-t", "--max-components-document", type=int, default=32,
                    help="the maximum number of the document-level components")
parser.add_argument("-n", "--num-epochs", type=int, default=101,
                    help="the number of epoches to pass through the data")
parser.add_argument("-m", "--batch-size", type=int, default=1024,
                    help="the number of the samples in the mini-batch")
parser.add_argument("-r", "--base-noise-ratio", type=float, default=1e-1,
                    help=("the (proportional) amount of uniform noise on the "
                          "inferred responsibility, for the further regularization"))
parser.add_argument("--whiten", type=bool, default=True,
                    action=argparse.BooleanOptionalAction,
                    help=("if set, feature dimension of the input sequences are "
                          "whitened using the precision matrix and mean vector."))
parser.add_argument("--share-alpha0", type=bool, default=False,
                    action=argparse.BooleanOptionalAction,
                    help=("set the model shares the DP prior ('alpha0') "
                          "across the whole corpus. Otherwise, each document "
                          "will have its own alpha0 value."))
parser.add_argument("--tau0", type=int, default=64,
                    help="learning rate decay offset")
parser.add_argument("--kappa", default=0.6,
                    type=partial(range_limited_float_type, min_val=.5, max_val=1.),
                    help="learning rate decay rate [0.5, 1)")
parser.add_argument("--batch-update", type=bool, default=False,
                    action=argparse.BooleanOptionalAction,
                    help=("if set as True, model update considering the full-batch "
                          "(not mini-batch)"))
parser.add_argument("--n-max-inner-iter", type=int, default=1000,
                    help=("the maximum number of inner inference loop per documents "
                          "within the mini-batch"))
parser.add_argument("--e-step-tol", type=float, default=1e-6,
                    help=("if the improvement of likelihood for current mini-batch "
                          "stops improving this factor (proportionally to the prev vlaue), "
                          "E-step inner loop terminates and move on the next mini-batch"))
parser.add_argument("--full-uniform-init", type=bool, default=False,
                    action=argparse.BooleanOptionalAction,
                    help=("if set, responsilibity initialization is using "
                          "the continuous uniform distribution, while it uses "
                          "discrete one through the hitogram."))
parser.add_argument("--max-len", type=int, default=2600,
                    help=("maximum length of sequence per mini-batch, "
                          "to save the memory, as current implementation is not "
                          "optimized with the high-variance lengths of sequences "
                          "within a mini-batch. The sequence longer than this value "
                          "will be cropped from the random location up to this length"))
parser.add_argument("--save-every", type=str, default='epoch',
                    help=("frequency for dumping checkpoints. If given as 'epoch', "
                          "it saves the model checkpoint end of every batch (epoch). "
                          "if given as integer, the script will parse and interprete it "
                          "as the number of mini-batch update for saving"))
parser.add_argument("--data-parallel-num-workers", type=int, default=1,
                    help="set the number of sub-process for handling data loading")
parser.add_argument("--prefix", type=str, default='',
                    help="model checkpoint filename prefix")
parser.add_argument("--device", type=str, default='cuda',
                    help="the main device that 'pytorch' uses to compute.")
parser.add_argument("--verbose", type=bool, default=True,
                    action=argparse.BooleanOptionalAction,
                    help="set verbosity")

args = parser.parse_args()


##########################
# default config template
##########################

save_every = (
    args.save_every
    if args.save_every == 'epoch' else
    int(args.save_every)
)

default = {
    'dataset': {
        'path': Path(args.train_data).resolve().as_posix(),
        'whiten': True
    },
    'model': {
        'max_components_corpus': args.max_components_corpus,
        'max_components_document': args.max_components_document,
        'n_epochs': args.num_epochs,
        'share_alpha0': args.share_alpha0,
        'tau0': args.tau0,
        'kappa': args.kappa,
        'batch_size': args.batch_size,
        'batch_update': args.batch_update,
        'n_max_inner_iter': args.n_max_inner_iter,
        'e_step_tol': args.e_step_tol,
        'base_noise_ratio': args.base_noise_ratio,
        'full_uniform_init': args.full_uniform_init,
        'max_len': args.max_len,
        'save_every': save_every,
        'data_parallel_num_workers': args.data_parallel_num_workers,
        'prefix': args.prefix,
        'out_path': Path(args.model_out).resolve().as_posix(),
        'device': args.device,  # we will experiment on nVidia GPUs
        'verbose': args.verbose
    }
}

############
# gen cases
############

configs = []
for noise_ratio in [0, 1e-4, 1e-3, 1e-2, 1e-1, 5e-1]:
    tmp_ = copy.deepcopy(default)
    tmp_['model']['base_noise_ratio'] = noise_ratio
    cur_prefix = tmp_['model']['prefix']
    tmp_['model']['prefix'] = f'{cur_prefix}{len(configs):d}'
    configs.append(tmp_)

# we'll do the rest once we find the best settings for the regularization
# computing the subset performance
# : 20k version
for i in range(5):
    noise_ratio = 1e-1
    tmp_ = copy.deepcopy(default)
    tmp_['model']['base_noise_ratio'] = noise_ratio
    cur_prefix = tmp_['model']['prefix']
    tmp_['model']['prefix'] = f'{cur_prefix}{len(configs):d}'

    tmp_['dataset']['path'] = (
        tmp_['dataset']['path']
        .replace('.h5', f'_subset20000_{i+1:d}.h5')
    )
    configs.append(tmp_)

# : 2k version
for i in range(5):
    noise_ratio = 1e-1
    tmp_ = copy.deepcopy(default)
    tmp_['model']['base_noise_ratio'] = noise_ratio
    cur_prefix = tmp_['model']['prefix']
    tmp_['model']['prefix'] = f'{cur_prefix}{len(configs):d}'

    tmp_['dataset']['path'] = (
        tmp_['dataset']['path']
        .replace('.h5', f'_subset2000_{i+1:d}.h5')
    )
    configs.append(tmp_)


# : short max length test
tmp_ = copy.deepcopy(default)
tmp_['model']['base_noise_ratio'] = 1e-1
cur_prefix = tmp_['model']['prefix']
tmp_['model']['prefix'] = f'{cur_prefix}{len(configs):d}'
tmp_['model']['max_len'] = 111  # corresponding to 2.677s with librosa default sr/nfft/hopsz
configs.append(tmp_)

# : K=512 test with batch_size=256
tmp_ = copy.deepcopy(default)
tmp_['model']['base_noise_ratio'] = 1e-1
tmp_['model']['max_components_corpus'] = 512
tmp_['model']['batch_size'] = 256
cur_prefix = tmp_['model']['prefix']
tmp_['model']['prefix'] = f'{cur_prefix}{len(configs):d}'
configs.append(tmp_)


#############
# save files
#############

out_path = Path(args.config_out)
out_path.mkdir(exist_ok=True, parents=True)
for i, config in enumerate(configs):
    out_fn = out_path / f'./{i:d}.json'
    with out_fn.open('w') as fp:
        json.dump(config, fp)
