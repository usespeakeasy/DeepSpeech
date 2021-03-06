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

# Create version_id for the run.
export VERSION_ID=$(openssl rand -hex 16)
echo $VERSION_ID

# original checkpoint shd be saved into /run/checkpoints/
# wget  https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-checkpoint.tar.gz
python -u DeepSpeech.py --noshow_progressbar \
  --train_files /data/training_features_all.csv \
  --test_files /data/test_features_all.csv \
  --dev_files /data/valid_features_all.csv \
  --train_batch_size 64 \
  --dev_batch_size 96 \
  --test_batch_size 96 \
  --learning_rate 0.0003 \
  --n_hidden 2048 \
  --train_cudnn \
  --epochs 30 \
  --export_tflite \
  --export_dir /run/model_export/ \
  --summary_dir /run/summaries/ \
  --checkpoint_dir /run/checkpoints/ \
  # --reduce_lr_on_plateau \
  # --plateau_epochs 2 \
  --test_output_file /run/test_output/test_output.json \
  "$@"

# copy model outputs to version bucket. 
gsutil -m cp -r /run/checkpoints/* gs://$GS_BUCKET_PATH/output/$VERSION_ID/checkpoints/
gsutil -m cp -r /run/model_export/* gs://$GS_BUCKET_PATH/output/$VERSION_ID/model_export/
gsutil -m cp -r /run/test_output/* gs://$GS_BUCKET_PATH/output/$VERSION_ID/test_output/

