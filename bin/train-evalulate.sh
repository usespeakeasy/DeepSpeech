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
  --train_batch_size 96 \
  --dev_batch_size 96 \
  --test_batch_size 96 \
  --learning_rate 0.0003 \
  --n_hidden 2048 \
  --train_cudnn \
  --epochs 60 \
  --export_tflite \
  --export_dir /run/model_export/ \
  --summary_dir /run/summaries/ \
  --checkpoint_dir /run/checkpoints/ \
  --test_output_file /run/test_output/test_output.json \
  "$@"

# copy model outputs to version bucket. 
gsutil -m cp -r /run/checkpoints/* gs://$GS_BUCKET_PATH/output/$VERSION_ID/checkpoints/
gsutil -m cp -r /run/model_export/* gs://$GS_BUCKET_PATH/output/$VERSION_ID/model_export/
gsutil -m cp -r /run/test_output/* gs://$GS_BUCKET_PATH/output/$VERSION_ID/test_output/

mkdir speak-data
cd /DeepSpeech/speak-data
gsutil -m cp gs://$GS_BUCKET_PATH/dataset/lm_text.txt .
mkdir speak-scorer


cd /DeepSpeech/data/lm
python generate_lm.py \
  --input_txt /DeepSpeech/speak-data/lm_text.txt \
  --output_dir /DeepSpeech/speak-data/speak-scorer \
  --top_k 500000 --kenlm_bins /DeepSpeech/native_client/kenlm/build/bin/ \
  --arpa_order 5 --max_arpa_memory "85%" --arpa_prune "0|0|1" \
  --binary_a_bits 255 --binary_q_bits 8 --binary_type trie

# generate scorer
curl -L https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/native_client.amd64.cuda.linux.tar.xz -o native_client.amd64.cuda.linux.tar.xz && tar -Jxvf native_client.amd64.cuda.linux.tar.xz

./generate_scorer_package \
  --alphabet ../alphabet.txt  \
  --lm ../../speak-data/speak-scorer/lm.binary \
  --vocab ../../speak-data/speak-scorer/vocab-500000.txt \
  --package kenlm-speak.scorer \
  --default_alpha 0.931289039105002 \
  --default_beta 1.1834137581510284

cp kenlm-speak.scorer ../../speak-data/speak-scorer/

# https://deepspeech.readthedocs.io/en/master/Scorer.html?highlight=language%20model#building-your-own-scorer
# TODO: hyperparameter search alpha - beta
cd /DeepSpeech/
python DeepSpeech.py \
  --test_files /data/test_features_all.csv \
  --checkpoint_dir /run/checkpoints/ \
  --export_dir /run/model_export/ \
  --n_hidden 2048 \
  --scorer /DeepSpeech/speak-data/speak-scorer/kenlm-speak.scorer

# generate scorer - again w/ tuned alpha and beta.
# cd /DeepSpeech/data/lm
# ./generate_scorer_package \
#   --alphabet ../alphabet.txt  \
#   --lm ../../speak-data/speak-scorer/lm.binary \
#   --vocab ../../speak-data/speak-scorer/vocab-500000.txt \
#   --package kenlm-speak.scorer \
#   --default_alpha $LM_ALPHA \
#   --default_beta $LM_BETA

cp kenlm-speak.scorer ../../speak-data/speak-scorer/
gsutil cp ../../speak-data/speak-scorer/* gs://$GS_BUCKET_PATH/output/$VERSION_ID/lm/