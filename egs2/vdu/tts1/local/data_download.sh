#!/usr/bin/env bash

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

download_dir=$1
corpus_file=$2

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 <download_dir>" "<corpus_file>"
    exit 1
fi

set -euo pipefail

cwd=$(pwd)
if [ ! -e "${download_dir}/corpus" ]; then
    mkdir -p "${download_dir}"
    cd "${download_dir}"
    # wget http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
    tar -vxf "${corpus_file}"
    mkdir -p "corpus/wavs"
    cd "WAV96CHUNK_RS"
    for f in *$'\226'*.wav; do
        base="${f##*$'\226'}"
        mv "$f" "../corpus/wavs/$base"
    done
    cd "${cwd}/${download_dir}"
    sed 's/^[^|]*–//' arn_transcripts.txt | sort > corpus/metadata.csv
    echo "successfully prepared data."
else    
    echo "already exists. skipped."
fi
