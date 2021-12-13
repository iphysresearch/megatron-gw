# coding=utf-8
# Copyright (c) 2021, PCL.  All rights reserved.
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

"""signal denoising evaluation, EVAL_ITER represents number of total samples in test dataset."""

from functools import partial

import torch
import torch.nn.functional as F

from megatron import get_args
from megatron import print_rank_0, print_rank_last
from megatron import get_timers
from megatron import mpu
from megatron.model import BertModel
from megatron.utils import average_losses_across_data_parallel_group


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building BERT model ...')

    args = get_args()
    num_tokentypes = 15 if args.bert_binary_head else 0

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

    keys = ['noisy_signal', 'clean_signal', 'params']
    datatype = torch.float64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    noisy_signal = data_b['noisy_signal']   #.long()
    clean_signal = data_b['clean_signal']   #.long()
    params = data_b['params']   #.long()

    return noisy_signal, clean_signal, params


# def loss_func(loss_mask, sentence_order, output_tensor):
def loss_func(loss_mask, sentence_order, clean_signal, output_tensor):

    denoised_signal, sop_logits = output_tensor
    # only calculate loss between denoised_data and clean_signal

    # lm_loss = torch.sum(torch.abs(denoised_signal-clean_signal).view(-1))
    loss_fn = torch.nn.L1Loss()
    lm_loss = loss_fn(denoised_signal, clean_signal)

    loss = lm_loss
    averaged_losses = average_losses_across_data_parallel_group(
        [lm_loss])
    return loss, {'lm loss': averaged_losses[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    noisy_signal, clean_signal, params = get_batch(data_iterator)

    # noisy_data and clean signal
    # loss_mask and sentence_order is used to calculate loss
    padding_mask = torch.ones(noisy_signal.shape[:2],device=noisy_signal.device)   # device='cuda:0'
    lm_labels = torch.ones(noisy_signal.shape[:2],device=noisy_signal.device) * -1
    types = torch.zeros(noisy_signal.shape[:2],device=noisy_signal.device)
    sentence_order = torch.zeros(noisy_signal.shape[0],device=noisy_signal.device)  # does not contribute to loss at demo stage
    loss_mask = torch.ones(noisy_signal.shape,device=noisy_signal.device)

    timers('batch-generator').stop()

    if not args.bert_binary_head:
        types = None

    # Forward pass through the model.
    output_tensor = model(noisy_signal, padding_mask, tokentype_ids=types,
                          lm_labels=lm_labels)

    if len(output_tensor)==2:
        import numpy as np
        vis_denoised = output_tensor[0].cpu().numpy()
        vis_noisy = noisy_signal.cpu().numpy()
        vis_clean = clean_signal.cpu().numpy()
        vis_all = np.append(vis_denoised, vis_noisy, axis=0)
        vis_all = np.append(vis_all, vis_clean, axis=0)
        from pathlib import Path
        import os
        folder = os.path.join('valid','{}_{}'.format(args.load.strip('/'), args.iteration))
        p = Path(folder)
        p.mkdir(parents=True, exist_ok=True)
        tmp_seed = args.consumed_valid_samples  #np.random.randint(10000)
        dprank = torch.distributed.get_rank()
        data_fn = 'data-{}-{}.npy'.format(dprank, tmp_seed)
        param_fn = 'param-{}-{}.npy'.format(dprank, tmp_seed)
        np.save(p / data_fn, vis_all)
        np.save(p / param_fn, params.cpu().numpy())


    # return output_tensor, partial(loss_func, loss_mask, sentence_order)
    return output_tensor, partial(loss_func, loss_mask, sentence_order, clean_signal)

def build_dataset(name, data_prefix, seed):
    from megatron.data.gw_dataset import GwDataset
    dataset = GwDataset(name=name,
                data_prefix=data_prefix[0],
                seed=seed)
    return dataset

def test_datasets_provider():
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for BERT ...')

    test_ds = build_dataset('test', data_prefix= args.data_path, seed=args.seed)
    # train_ds = None
    # valid_ds = None

    # return train_ds, valid_ds, test_ds
    return test_ds

import time
_TRAIN_START_TIME = time.time()
from datetime import datetime
from megatron.initialize import initialize_megatron
from megatron.training import setup_model_and_optimizer #, evaluate_and_print_results
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron import get_tensorboard_writer, is_last_rank, get_num_microbatches
from megatron.schedules import get_forward_backward_func
import math

def evaluate(forward_step_func, data_iterator, model, verbose=False):
    """Evaluation."""
    args = get_args()

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    total_loss_dict = {}

    with torch.no_grad():
        iteration = 0
        args.consumed_valid_samples = 0
        while args.consumed_valid_samples + args.global_batch_size <= args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration,
                                                            args.eval_iters))

            forward_backward_func = get_forward_backward_func()
            loss_dicts = forward_backward_func(
                forward_step_func, data_iterator, model, optimizer=None,
                timers=None, forward_only=True)

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # Reduce across processes.
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        total_loss_dict[key] = total_loss_dict.get(
                            key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]

            args.consumed_valid_samples += mpu.get_data_parallel_world_size() \
                                           * args.micro_batch_size \
                                           * get_num_microbatches()
    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= args.eval_iters * get_num_microbatches()

    return total_loss_dict

def evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, model,
                               iteration, verbose=False):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    writer = get_tensorboard_writer()

    total_loss_dict = evaluate(forward_step_func, data_iterator, model, verbose)
    string = ' validation loss at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer and is_last_rank():
            writer.add_scalar('{} validation'.format(key),
                              total_loss_dict[key].item(),
                              iteration)
            writer.add_scalar('{} validation vs samples'.format(key),
                              total_loss_dict[key].item(),
                              args.consumed_train_samples)
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar('{} validation ppl'.format(key), ppl,
                                  iteration)
                writer.add_scalar('{} validation ppl vs samples'.format(key),
                                  ppl, args.consumed_train_samples)

    length = len(string) + 1
    print_rank_last('-' * length)
    print_rank_last(string)
    print_rank_last('-' * length)

def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))

def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x

def build_test_data_iterators(
        build_test_datasets_provider):
    """XXX"""
    args = get_args()

    test_dataloader = None

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_tensor_model_parallel_rank() == 0:

        # Build the datasets.
        test_ds = build_test_datasets_provider()

        # Build dataloders.
        test_dataloader = build_pretraining_data_loader(test_ds, 0)

        # Flags to know if we need to do training/validation/testing.
        do_test = test_dataloader is not None and args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor(
            [0, 0, int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(flags,
                                mpu.get_tensor_model_parallel_src_rank(),
                                group=mpu.get_tensor_model_parallel_group())
    args.do_train = flags[0].item()
    args.do_valid = flags[1].item()
    args.do_test = flags[2].item()

    # Build iterators.
    dl_type = args.dataloader_type
    assert dl_type in ['single', 'cyclic']

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader) if dl_type == 'single' \
                             else iter(cyclic_iter(test_dataloader))
    else:
        test_data_iterator = None

    return test_data_iterator

def test(test_dataset_provider,
             model_provider,
             forward_step_func,
             extra_args_provider=None,
             args_defaults={}):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Arguments:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.cuda.FloatTensor([_TRAIN_START_TIME])
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')

    args = get_args()
    timers = get_timers()

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup').start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)
    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')

    # Data stuff.
    timers('test-data-iterators-setup').start()
    if args.virtual_pipeline_model_parallel_size is not None:
        all_data_iterators = [
            build_test_data_iterators(test_dataset_provider)
            for _ in range(len(model))
        ]
        test_data_iterator = [data_iterators[0] for data_iterators in all_data_iterators]
    else:
        test_data_iterator = build_test_data_iterators(test_dataset_provider)
    timers('test-data-iterators-setup').stop()
    print_datetime('after dataloaders are built')

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup', 'test-data-iterators-setup'])
    print_rank_0('testing ...')

    print_datetime('start testing')
    if args.do_test:
        # Run on test data.
        prefix = 'the end of training for test data'
        evaluate_and_print_results(prefix, forward_step_func,
                                   test_data_iterator, model,
                                   0, True)
    print_datetime('after tests are done')

if __name__ == "__main__":

    test(test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
