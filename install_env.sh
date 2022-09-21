#!/bin/bash

# some functions that we need...

error_exit() {
    # throw an "exception" with message
    #
    # $1 : error message that will be thrown
    #

    echo "[ERROR] $1"
    exit 1
}


download_url () {
    # download file from url input and save it into path
    #
    # $1 : target url to fetch
    # $2 : output filename
    #

    if command -v wget &> /dev/null ; then
        wget -O $2 $1
    elif command -v curl &> /dev/null ; then
        curl -o $2 -L -O $1
    else
        error_exit "curl and wget are not available. install either of them to proceed"
    fi
}



# check os types and throw error if it's not supported
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    miniconda_url=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
elif [[ "$OSTYPE" == "darwin"* ]]; then
    miniconda_url=https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
else
    # we currently only support macos and linux
    error_exit "We currently only support linux and mac!"
fi


# set some variables
frozen_env=environment.yaml


# intall miniconda
mkdir -p ./miniconda3
download_url $miniconda_url ./miniconda3/miniconda.sh
bash ./miniconda3/miniconda.sh -b -u -p ./miniconda3
rm -rf ./miniconda3/miniconda.sh


# install environment
./miniconda3/bin/conda env create -f $frozen_env
source ./miniconda3/bin/activate hdpgmm_music_experiment
