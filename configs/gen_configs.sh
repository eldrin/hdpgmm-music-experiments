#!/bin/bash

PROJ_ROOT=/tudelft.net/staff-umbrella/expmusicrec
HDPGMM_MSD_TRAIN="${PROJ_ROOT}/datasets/msd200k_features.h5"
HDPGMM_OUT_PATH="${PROJ_ROOT}/preojects/music_replrn_hdp/data/models"
CONFIG_ROOT="${PROJ_ROOT}/projects/music_replrn_hdp/configs"

python "${CONFIG_ROOT}/generate_configs.py" \
    $CONFIG_ROOT \
    $HDPGMM_MSD_TRAIN \
    $HDPGMM_OUT_PATH
