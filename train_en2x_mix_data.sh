#!/usr/bin/env bash

# TGT_LANG=$1
# MODEL_DIR=$2
# shift 1

# download Wav2vec2 model

# mkdir -p checkpoints
# wget -P checkpoints https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt

# mkdir -p ${MODEL_DIR}
# nohup bash train_en2x.sh >> const_withword.txt &
CUDA_VISIBLE_DEVICES=0 python train.py /data/zxh/st/STEMM/data/mustc \
    --task speech_to_text_triplet_with_extra_mt_mix \
    --train-subset train_stemm_const2 --valid-subset dev_st \
    --config-yaml config_st.yaml \
    --langpairs en-de --lang-prefix-tok "<lang:de>" \
    --max-audio-positions 600000 --max-source-positions 1024 --max-target-positions 1024 \
    --max-audio-tokens 1000000 --max-text-tokens 2000 --max-tokens 1000000  --max-tokens-valid 2000000 \
    --skip-invalid-size-inputs-valid-test \
    --text-data-sample-ratio 0.25 \
    --mix-rate 0.6 \
    --word-dict-path /data/zxh/st/STEMM/data/mustc/const_word_dict.tsv \
    --tensorboard-logdir "./logs" \
    \
    --arch xstnet_base --w2v2-model-path /data/zxh/st/STEMM/checkpoints/wav2vec_small.pt \
    \
    --optimizer adam --clip-norm 10.0 \
    --lr-scheduler inverse_sqrt --lr 1e-4  --warmup-updates 25000  --weight-decay 0.0 \
    \
    --criterion multi_task_cross_entropy_with_contrastive_with_extra_MT \
    --label-smoothing 0.1 --ignore-prefix-size 1 --report-accuracy \
    --contrastive-weight 0 --contrastive-temperature 0.02 --contrastive-seqlen-type none \
    \
    --update-freq 1 --max-update 500000 \
    \
    --no-progress-bar --log-format json --log-interval 100 \
    --save-interval-updates 1000 --save-interval 1 \
    --keep-last-epochs 10 --keep-interval-updates 15 --keep-best-checkpoints 10 \
    --save-dir /data/zxh/st/ConST/model/test \
    --ddp-backend=no_c10d --fp16 \
    \
    --eval-bleu --eval-bleu-args '{"beam": 4, "prefix_size": 1}' \
    --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples \
    --eval-bleu-bpe sentencepiece --eval-bleu-bpe-path ./data/spm_unigram10000_st.model \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric