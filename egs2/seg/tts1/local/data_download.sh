
download_dir=$1
corpus_file=$2

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 <download_dir>" "<corpus_file>"
    exit 1
fi

set -euo pipefail

cwd=$(pwd)
if [ ! -e "${download_dir}/corpus/.done" ]; then
    mkdir -p "${download_dir}/corpus"
    unzip "$corpus_file" -d ${download_dir}/corpus
    echo "successfully extracted data."
    touch "${download_dir}/corpus/.done"
else    
    echo "already exists. skipped."
fi
