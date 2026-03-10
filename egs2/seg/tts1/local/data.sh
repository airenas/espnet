#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=-1
stop_stage=2

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ -z "${corpus_file}" ]; then
   log "Fill the env value of 'corpus_file'"
   exit 1
fi
if [ -z "${work_dir}" ]; then
   log "Fill the env value of 'work_dir'"
   exit 1
fi
db_root="$(realpath ${work_dir})/downloads"

log "Starting work on: ${db_root}"

train_set=tr_no_dev
train_dev=dev
eval_set=eval1

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Data Download"
    local/data_download.sh "${db_root}" "${corpus_file}"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data Preparation"
    # set filenames
    scp=${work_dir}/data/train/wav.scp
    utt2spk=${work_dir}/data/train/utt2spk
    spk2utt=${work_dir}/data/train/spk2utt
    text=${work_dir}/data/train/text
    durations=${work_dir}/data/train/durations
    # check file existence
    [ ! -e ${work_dir}/data/train ] && mkdir -p ${work_dir}/data/train
    [ -e ${scp} ] && rm ${scp}
    [ -e ${utt2spk} ] && rm ${utt2spk}
    [ -e ${spk2utt} ] && rm ${spk2utt}
    [ -e ${text} ] && rm ${text}
    [ -e ${durations} ] && rm ${durations}

    wavs_dir="${db_root}/corpus/wavs"
    # make scp, utt2spk, and spk2utt
    find "${wavs_dir}" -name "*.wav" | sort | while read -r filename; do
        id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g")
        echo "${id} ${filename}" >> ${scp}
        echo "${id} ${speaker}" >> ${utt2spk}
    done
    utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}

    # make text using the original text
    # cleaning and phoneme conversion are performed on-the-fly during the training
    paste -d " " \
        <(cut -d "|" -f 1 < ${db_root}/corpus/metadata.csv) \
        <(cut -d "|" -f 3 < ${db_root}/corpus/metadata.csv) \
        > ${text}
    sort ${text} -o ${text}

    utils/validate_data_dir.sh --no-feats ${work_dir}/data/train
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: utils/subset_data_dir.sh"
    log "Making subsets for training and evaluation. ${dev_count} + ${dev_count} utterances are used for dev eval and the rest for training."
    deveval_count=$((dev_count * 2))
    # make evaluation and development sets
    utils/subset_data_dir.sh --last ${work_dir}/data/train ${deveval_count} ${work_dir}/data/deveval
    utils/subset_data_dir.sh --last ${work_dir}/data/deveval ${dev_count} ${work_dir}/data/${eval_set}
    utils/subset_data_dir.sh --first ${work_dir}/data/deveval ${dev_count} ${work_dir}/data/${train_dev}
    n=$(( $(wc -l < ${work_dir}/data/train/wav.scp) - ${deveval_count} ))
    utils/subset_data_dir.sh --first ${work_dir}/data/train ${n} ${work_dir}/data/${train_set}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
