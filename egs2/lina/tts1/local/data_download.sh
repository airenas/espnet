#!/bin/bash

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

download_dir=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <download_dir>"
    exit 1
fi

set -euo pipefail

cwd=$(pwd)
if [ ! -e "${download_dir}/${corpus}" ]; then
    echo "please prepare audio corpus in ${download_dir}/${corpus}"
    exit 1
else
    echo "already exists. skipped."
fi
