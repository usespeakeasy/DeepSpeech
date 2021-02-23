#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

mkdir -p /run/summaries/
mkdir -p /run/checkpoints/
mkdir -p /run/test_output/
mkdir -p /run/model_export/

# https://github.com/mozilla/DeepSpeech/issues/3088
export TF_CUDNN_RESET_RND_GEN_STATE=1 

# original checkpoint shd be saved into /run/checkpoints/
# wget  https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-checkpoint.tar.gz
python -u DeepSpeech.py --noshow_progressbar \
  --train_files /data/training_features.csv \
  --test_files /data/test_features.csv \
  --dev_files /data/valid_features.csv \
  --train_batch_size 32 \
  --dev_batch_size 32 \
  --test_batch_size 32 \
  --learning_rate 0.00005 \
  --n_hidden 2048 \
  --train_cudnn \
  --epochs 20 \
  --export_tflite \
  --export_dir /run/model_export/ \
  --summary_dir /run/summaries/ \
  --checkpoint_dir /run/checkpoints/ \
  --reduce_lr_on_plateau \
  --plateau_epochs 2 \
  --test_output_file /run/test_output/test_output.json \
  "$@"
