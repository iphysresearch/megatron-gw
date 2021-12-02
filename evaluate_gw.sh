#!/bin/bash

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=192.168.202.149
MASTER_PORT=6066 # 6006
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=./bigdata/ #./gwdata/
CHECKPOINT_PATH=gwdmp  #pretrain-bert
VOCAB_FILE=./bert_vocab_files/bert-base-uncased-vocab.txt

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

export OMP_NUM_THREADS=12
#export NCCL_DEBUG=INFO
# CUDA_LAUNCH_BLOCKING=1 
export CUDA_VISIBLE_DEVICES=1,2,3,4
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       evaluate_gw.py \
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 2048 \
       --num-attention-heads 32 \
       --micro-batch-size 4 \
       --global-batch-size 16 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 100000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --lr-decay-iters 99000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .002 \
       --log-interval 100 \
       --save-interval 100000 \
       --eval-interval 1000 \
       --eval-iters 1 \
       --dataloader-type cyclic \
       --fp16
#       --global-batch-size 32 \

