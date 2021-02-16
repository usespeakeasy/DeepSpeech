#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;



if [ -d "${COMPUTE_KEEP_DIR}" ]; then
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/ldc93s1"))')
fi


python -u DeepSpeech.py --noshow_progressbar \
  --train_files /data/training_features.csv \
  --test_files data/test_features.csv \
  --dev_files data/valid_features.csv \
  --train_batch_size 8 \
  --dev_batch_size 8 \
  --test_batch_size 8 \
  --n_hidden 100 \
  --epochs 1 \
  --export_tflite
  --checkpoint_dir "$checkpoint_dir" \
  "$@"
