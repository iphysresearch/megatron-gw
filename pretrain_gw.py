# coding=utf-8
# Copyright (c) 2020, PCL.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain BERT"""

from functools import partial

import torch
import torch.nn.functional as F

from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import mpu
from megatron.data.dataset_utils import build_train_valid_test_datasets
from megatron.model import BertModel
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building BERT model ...')

    args = get_args()
    num_tokentypes = 15 if args.bert_binary_head else 0
#    model = BertModel(
#        num_tokentypes=num_tokentypes,
#        add_binary_head=args.bert_binary_head,
#        parallel_output=True,
#        pre_process=pre_process,
#        post_process=post_process)

    model = BertModel(
        num_tokentypes=num_tokentypes,
        add_binary_head=args.bert_binary_head,
        parallel_output=False,
        pre_process=pre_process,
        post_process=post_process)

    return model


def get_batch(data_iterator):
    """Build the batch."""

    # Items and their type.
    # keys = ['text', 'types', 'labels', 'is_random', 'loss_mask', 'padding_mask']
    # datatype = torch.int64

    keys = ['noisy_signal', 'clean_signal', 'mask', 'params']
    datatype = torch.float64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    noisy_signal = data_b['noisy_signal'] #.long()
    clean_signal = data_b['clean_signal'] #.long()
    loss_mask = data_b['mask'] #.long()
    params = data_b['params'] #.long()

    return noisy_signal, clean_signal, loss_mask, params


# def loss_func(loss_mask, sentence_order, output_tensor):
def loss_func(loss_mask, sentence_order, clean_signal, output_tensor):

    denoised_signal, sop_logits = output_tensor
    # only calculate loss between denoised_data and clean_signal
    #lm_loss = torch.sum(torch.abs(denoised_signal-clean_signal).view(-1))
    #loss_fn = torch.nn.L1Loss()
    #lm_loss = loss_fn(denoised_signal, clean_signal)
    loss_fn = torch.nn.MSELoss()
    #lm_loss = loss_fn(denoised_signal.to(torch.float32), clean_signal.to(torch.float32))
    lm_loss = loss_fn(denoised_signal.to(torch.float32) * loss_mask.to(torch.float32), clean_signal.to(torch.float32) * loss_mask.to(torch.float32))

    loss = lm_loss
    averaged_losses = average_losses_across_data_parallel_group(
        [lm_loss])
    return loss, {'lm loss': averaged_losses[0]}

    # lm_loss_, sop_logits = output_tensor
    #
    # lm_loss_ = lm_loss_.float()
    # loss_mask = loss_mask.float()
    # lm_loss = torch.sum(
    #     lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()
    #
    # if sop_logits is not None:  # [batch size, 2]
    #     sop_loss = F.cross_entropy(sop_logits.view(-1, 2).float(),
    #                                sentence_order.view(-1),
    #                                ignore_index=-1)
    #     sop_loss = sop_loss.float()
    #     loss = lm_loss + sop_loss
    #     averaged_losses = average_losses_across_data_parallel_group(
    #         [lm_loss, sop_loss])
    #     return loss, {'lm loss': averaged_losses[0],
    #                   'sop loss': averaged_losses[1]}
    #
    # else:
    #     loss = lm_loss
    #     averaged_losses = average_losses_across_data_parallel_group(
    #         [lm_loss])
    #     return loss, {'lm loss': averaged_losses[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    # noisy_signal, clean_signal, params = get_batch(data_iterator)
    noisy_signal, clean_signal, loss_mask, params = get_batch(data_iterator)

    # noisy_data and clean signal
    # loss_mask and sentence_order is used to calculate loss
    padding_mask = torch.ones(noisy_signal.shape[:2],device=noisy_signal.device)   # device='cuda:0'
    lm_labels = torch.ones(noisy_signal.shape[:2],device=noisy_signal.device) * -1
    types = torch.zeros(noisy_signal.shape[:2],device=noisy_signal.device)
    sentence_order = torch.zeros(noisy_signal.shape[0],device=noisy_signal.device)  # does not contribute to loss at demo stage
    # loss_mask = torch.ones(noisy_signal.shape,device=noisy_signal.device)

    timers('batch-generator').stop()

    if not args.bert_binary_head:
        types = None

    # Forward pass through the model.
    output_tensor = model(noisy_signal, padding_mask, tokentype_ids=types,
                          lm_labels=lm_labels)

    # if torch.distributed.get_rank() == 0:
    #     import numpy as np
    #     np.save('./realtest/noisy_signal_input.npy', noisy_signal.cpu().numpy())
    #     np.save('./realtest/noisy_signal_float16.npy', noisy_signal.to(args.params_dtype).cpu().numpy())
    #     np.save('./realtest/clean_signal_label.npy', clean_signal.cpu().numpy())
    #     np.save('./realtest/clean_signal_float32.npy', clean_signal.to(torch.float32).cpu().numpy())
    #     np.save('./realtest/denoised_signal_output.npy', output_tensor[0].detach().cpu().numpy())
    #     np.save('./realtest/denoised_signal_float32.npy', output_tensor[0].to(torch.float32).detach().cpu().numpy())
    #     np.save('./realtest/loss_mask.npy', loss_mask.to(torch.float32).cpu().numpy())

    # return output_tensor, partial(loss_func, loss_mask, sentence_order)
    # print("loss mask shape:{}".format(loss_mask.shape))
    return output_tensor, partial(loss_func, loss_mask, sentence_order, clean_signal)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for BERT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        max_seq_length=args.seq_length,
        masked_lm_prob=args.mask_prob,
        short_seq_prob=args.short_seq_prob,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        binary_head=args.bert_binary_head)
    print_rank_0("> finished creating BERT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})

