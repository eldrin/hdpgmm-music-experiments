from pathlib import Path
import argparse
import os
import json


############
# Argparser
############

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
args = parser.parse_args()


##########################
# default config template
##########################

default = {
    'dataset': {
        'path': args.train_data,
        'whiten': True
    },
    'model': {
        'max_components_corpus': 256,
        'max_components_document': 32,
        'n_epochs': 101,
        'share_alpha0': False,
        'tau0': 64,
        'kappa': 0.6,
        'batch_size': 1024,
        'batch_update': False,
        'n_max_inner_iter': 100,
        'e_step_tol': 1e-4,
        'base_noise_ratio': 1e-1,
        'full_uniform_init': False,
        'max_len': 2600,
        'save_every': 'epoch',
        'data_parallel_num_workers': 1,
        'prefix': '',
        'out_path': args.model_out,
        'device': 'cuda',  # we will experiment on nVidia GPUs
        'verbose': True
    }
}

############
# gen cases
############

configs = []
for noise_ratio in [0, 1e-4, 1e-3, 1e-2, 1e-1]:
    tmp_ = default.copy()
    tmp_['noise_ratio'] = noise_ratio
    tmp_['prefix'] = f'{len(configs):d}'
    configs.append(tmp_)

# we'll do the rest once we find the best settings for the regularization

#############
# save files
#############

out_path = Path(args.config_out)
out_path.mkdir(exist_ok=True, parents=True)
for i, config in enumerate(configs):
    out_fn = out_path / f'./{i:d}.json'
    with out_fn.open('w') as fp:
        json.dump(config, fp)
