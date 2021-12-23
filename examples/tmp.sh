#!/bin/bash

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=192.168.202.149
MASTER_PORT=7456 # 6006
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=../bigdata/
CHECKPOINT_PATH=verify1  #pretrain-bert
VOCAB_FILE=./bert_vocab_files/bert-base-uncased-vocab.txt

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

#export OMP_NUM_THREADS=12
#export NCCL_DEBUG=INFO
# CUDA_LAUNCH_BLOCKING=1 
#export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gw.py \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 2048 \
       --num-attention-heads 32 \
       --micro-batch-size 4 \
       --global-batch-size 32 \
       --segment-length 2048 \
       --seq-length 127 \
       --max-position-embeddings 127 \
       --train-iters 200000 \
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
       --save-interval 20000 \
       --eval-interval 500 \
       --eval-iters 10 \
       --dataloader-type cyclic \
       --fp16 \
       --bert-no-binary-head
#       --global-batch-size 32 \
#       --num-layers-per-virtual-pipeline-stage 3 \
