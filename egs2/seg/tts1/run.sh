#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=22050
n_fft=1024
n_shift=256

cleaner=${cleaner}
token_type=${token_type}
# g2p=espeak_ng_lt

# cleaner=lower
# token_type=char

opts=
if [ "${fs}" -eq 22050 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

train_set=tr_no_dev
valid_set=dev
test_sets="dev eval1"

./tts.sh \
    --lang lt \
    --feats_type raw \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --token_type "${token_type}" \
    --cleaner "${cleaner}" \
    --g2p "${g2p}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "${work_dir}/data/${train_set}/text" \
    --f0min ${f0min} --f0max ${f0max} \
    ${opts} "$@"
