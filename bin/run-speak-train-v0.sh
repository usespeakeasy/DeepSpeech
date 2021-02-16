#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

mkdir -p /run/summaries/
mkdir -p /run/checkpoints/
mkdir -p /run/test_output_files/
mkdir -p /run/model_export/

python -u DeepSpeech.py --noshow_progressbar \
  --train_files /data/training_features.csv \
  --test_files /data/test_features.csv \
  --dev_files /data/valid_features.csv \
  --train_batch_size 16 \
  --dev_batch_size 16 \
  --test_batch_size 16 \
  --n_hidden 100 \
  --epochs 10 \
  --export_tflite \
  --export_dir /run/model_export/ \
  --summary_dir /run/summaries/ \
  --checkpoint_dir /run/checkpoints/ \
  --test_output_file /run/test_output_files/test_output_file.json \
  "$@"
