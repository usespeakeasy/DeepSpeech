#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

mkdir -p ../summaries
mkdir -p ../checkpoints

python -u DeepSpeech.py --noshow_progressbar \
  --train_files /data/training_features.csv \
  --test_files /data/test_features.csv \
  --dev_files /data/valid_features.csv \
  --train_batch_size 8 \
  --dev_batch_size 8 \
  --test_batch_size 8 \
  --n_hidden 100 \
  --epochs 1 \
  --export_tflite
  --summary_dir "../summaries"
  --checkpoint_dir "../checkpoints" \
  "$@"
