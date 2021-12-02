#!/bin/bash

RANK=0
WORLD_SIZE=1
DATA_PATH=./one/my-bert_text_sentence
CHECKPOINT_PATH=datadebug
export CUDA_VISIBLE_DEVICES=0

python -m pdb pretrain_bert.py \
       --num-layers 24 \
       --hidden-size 2048 \
       --num-attention-heads 32 \
       --micro-batch-size 1 \
       --global-batch-size 1 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 20000 \
       --lr-decay-iters 9900 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file ./bert_vocab_files/bert-base-uncased-vocab.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --lr 0.0001 \
       --min-lr 0.00001 \
       --lr-decay-style linear \
       --lr-warmup-fraction .01 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16
