#!/usr/bin/env python
#
# This code is based on CEDR: https://github.com/Georgetown-IR-Lab/cedr
# It has some modifications/extensions and it relies on our custom BERT
# library: https://github.com/searchivarius/pytorch-pretrained-BERT-mod
# (c) Georgetown IR lab & Carnegie Mellon University
# It's distributed under the MIT License
# MIT License is compatible with Apache 2 license for the code in this repo.
#
import os
import time
import gc
import sys
import math
import argparse
import torch.distributed as dist


import flexneuart.models.train.data as data

from flexneuart.models.utils import add_model_init_basic_args

from flexneuart.models.base import ModelSerializer, MODEL_PARAM_PREF
from flexneuart.models.train.loss import *
from flexneuart.models.train.amp import *
from flexneuart.models.train.data import QUERY_ID_FIELD, DOC_ID_FIELD, CAND_SCORE_FIELD, \
                                    DOC_TOK_FIELD, DOC_MASK_FIELD, \
                                    QUERY_TOK_FIELD, QUERY_MASK_FIELD

from flexneuart import sync_out_streams, set_all_seeds, join_and_check_stat, enable_spawn
from flexneuart.io.json import read_json, save_json
from flexneuart.io.runs import read_run_dict
from flexneuart.io.qrels import read_qrels_dict
from flexneuart.eval import METRIC_LIST, get_eval_results

from flexneuart.config import DEVICE_CPU, TQDM_FILE, PYTORCH_DISTR_BACKEND

from tqdm import tqdm
from collections import namedtuple
from multiprocessing import Process
from threading import BrokenBarrierError
from multiprocessing import Barrier

# 20 minutes should be more than enough while waiting
# for other processes to reach the same training point
BARRIER_WAIT_MODEL_AVERAGE_TIMEOUT=60*20
# However (see comment below) we should wait more before validation completes.
# Let's some optimisticially assume, it is not longer than 24 hours,
# This needs to be fixed in the future:
# A good fix should make validation use all GPUs.
BARRIER_WAIT_VALIDATION_TIMEOUT=3600 * 24

OPT_SGD='sgd'
OPT_ADAMW='adamw'

VALID_ALWAYS = 'always'
VALID_LAST = 'last_epoch'
VALID_NONE = 'never'

TrainParams = namedtuple('TrainParams',
                    ['optim',
                     'init_lr', 'init_bert_lr', 'epoch_lr_decay', 'weight_decay',
                     'momentum',
                     'amp',
                     'warmup_pct', 'batch_sync_qty',
                     'batches_per_train_epoch',
                     'batch_size', 'batch_size_val',
                     'max_query_len', 'max_doc_len',
                     'cand_score_weight', 'neg_qty_per_query',
                     'backprop_batch_size',
                     'epoch_qty',
                     'save_epoch_snapshots', 'save_last_snapshot_every_k_batch',
                     'device_name', 'print_grads',
                     'shuffle_train',
                     'valid_type',
                     'use_external_eval', 'eval_metric'])


def avg_model_params(model, amp):
    """
       Average model parameters across all GPUs. 
       Set amp to True, to enable automatic mixed-precision.
    """
    auto_cast_class, scaler = get_amp_processors(amp)

    with auto_cast_class():
        qty = float(dist.get_world_size())
        for prm in model.parameters():
            dist.all_reduce(prm.data, op=torch.distributed.ReduceOp.SUM)
            prm.data /= qty

def clean_memory(device_name):
    sync_out_streams()
    print('\n', 'Clearning memory device:', device_name)
    sync_out_streams()
    gc.collect()
    if device_name != DEVICE_CPU:
        with torch.cuda.device(device_name):
            torch.cuda.empty_cache()


def get_lr_desc(optimizer):
    lr_arr = ['LRs:']
    for param_group in optimizer.param_groups:
        lr_arr.append('%.6f' % param_group['lr'])

    return ' '.join(lr_arr)


class ValidationTimer:
    def __init__(self, validation_checkpoints):
        self.validation_checkpoints = sorted(validation_checkpoints)
        self.pointer = 0
        self.total_steps = 0

    def is_time(self):
        if self.pointer >= len(self.validation_checkpoints):
            return False
        if self.total_steps >= self.validation_checkpoints[self.pointer]:
            self.pointer += 1
            return True
        return False

    def last_checkpoint(self):
        return self.validation_checkpoints[self.pointer - 1]

    def increment(self, steps_qty):
        self.total_steps += steps_qty


def train_iteration(model, sync_barrier,
                    is_master_proc, device_qty,
                    loss_obj,
                    train_params, max_train_qty,
                    valid_run, valid_qrel_filename,
                    optimizer, scheduler,
                    dataset, train_pairs, qrels,
                    validation_timer, valid_run_dir, valid_scores_holder,
                    save_last_snapshot_every_k_batch,
                    model_out_dir):

    clean_memory(train_params.device_name)

    model.train()
    total_loss = 0.
    total_prev_qty = total_qty = 0. # This is a total number of records processed, it can be different from
                                    # the total number of training pairs

    batch_size = train_params.batch_size

    optimizer.zero_grad()

    if train_params.print_grads:
      print('Gradient sums before training')
      for k, v in model.named_parameters():
        print(k, 'None' if v.grad is None else torch.sum(torch.norm(v.grad, dim=-1, p=2)))

    lr_desc = get_lr_desc(optimizer)

    batch_id = 0
    snap_id = 0

    if is_master_proc:

        sync_out_streams()

        pbar = tqdm('training', total=max_train_qty, ncols=80, desc=None, leave=False, file=TQDM_FILE)
    else:
        pbar = None

    if loss_obj.is_listwise():
        neg_qty_per_query = train_params.neg_qty_per_query
        assert neg_qty_per_query >= 1
    else:
        neg_qty_per_query = 1

    cand_score_weight = torch.FloatTensor([train_params.cand_score_weight]).to(train_params.device_name)

    auto_cast_class, scaler = get_amp_processors(train_params.amp)

    for record in data.iter_train_data(model, train_params.device_name, dataset,
                                       train_pairs,
                                       train_params.shuffle_train, neg_qty_per_query,
                                       qrels, train_params.backprop_batch_size,
                                       train_params.max_query_len, train_params.max_doc_len):

        data_qty = len(record['query_id'])
        with auto_cast_class():
            scores = model(record[QUERY_TOK_FIELD],
                       record[QUERY_MASK_FIELD],
                       record[DOC_TOK_FIELD],
                       record[DOC_MASK_FIELD]) + record[CAND_SCORE_FIELD] * cand_score_weight

            # +1 b/c one score is for the positive document
            count = data_qty // (neg_qty_per_query + 1)
            assert count * (neg_qty_per_query + 1) == data_qty
            scores = scores.reshape(count, 1 + neg_qty_per_query)
            loss = loss_obj.compute(scores)

        scaler.scale(loss).backward()
        total_qty += count

        if train_params.print_grads:
          print(f'Records processed {total_qty} Gradient sums:')
          for k, v in model.named_parameters():
            print(k, 'None' if v.grad is None else torch.sum(torch.norm(v.grad, dim=-1, p=2)))

        total_loss += loss.item()

        if is_master_proc:
            validation_timer.increment(1)

        run_chkpt_val = is_master_proc and validation_timer.is_time() and valid_run_dir is not None

        # If it's time to validate, we need to interrupt the batch
        if total_qty - total_prev_qty >= batch_size or run_chkpt_val:

            #optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()
            total_prev_qty = total_qty

            # Scheduler must make a step in each batch! *AFTER* the optimizer makes an update!
            if scheduler is not None:
                scheduler.step()
                lr_desc = get_lr_desc(optimizer)

            # This must be done in every process, not only in the master process
            if device_qty > 1:
                if batch_id % train_params.batch_sync_qty == 0:
                    try:
                        sync_barrier.wait(BARRIER_WAIT_MODEL_AVERAGE_TIMEOUT)
                    except BrokenBarrierError:
                        raise Exception('A waiting-for-model-parameter-synchronization timeout!')

                    avg_model_params(model, train_params.amp)

            batch_id += 1

            # We will surely skip batch_id == 0
            if save_last_snapshot_every_k_batch is not None and batch_id % save_last_snapshot_every_k_batch == 0:
                if is_master_proc:
                    os.makedirs(model_out_dir, exist_ok=True)
                    out_tmp = os.path.join(model_out_dir, f'model.last.{snap_id}')
                    model.save_all(out_tmp)
                    snap_id += 1

        if pbar is not None:
            pbar.update(count)
            pbar.refresh()
            sync_out_streams()
            pbar.set_description('%s train loss %.5f' % (lr_desc, total_loss / float(total_qty)) )

        while run_chkpt_val:
            model.eval()
            os.makedirs(valid_run_dir, exist_ok=True)
            run_file_name = os.path.join(valid_run_dir, f'batch_{validation_timer.last_checkpoint()}.run')
            pbar.refresh()
            sync_out_streams()
            score = validate(model, train_params, dataset,
                             valid_run,
                             qrelf=valid_qrel_filename, run_filename=run_file_name)

            pbar.refresh()
            sync_out_streams()
            pbar.write(f'\n# of steps={validation_timer.total_steps} score={score:.4g}\n')
            valid_scores_holder[f'batch_{validation_timer.last_checkpoint()}'] = score
            save_json(os.path.join(valid_run_dir, "scores.json"), valid_scores_holder)
            model.train()
            # We may need to make more than one validation iteration
            run_chkpt_val = run_chkpt_val = is_master_proc and validation_timer.is_time() and valid_run_dir is not None

        if total_qty >= max_train_qty:
            break

    # Final model averaging in the end.

    if device_qty > 1:
        try:
            sync_barrier.wait(BARRIER_WAIT_MODEL_AVERAGE_TIMEOUT)
        except BrokenBarrierError:
            raise Exception('A waiting-for-model-parameter-synchronization (in the end of epoch) timeout!')

        avg_model_params(model, train_params.amp)

    if pbar is not None:
        pbar.close()
        sync_out_streams()

    return total_loss / float(total_qty)


def validate(model, train_params, dataset, orig_run, qrelf, run_filename):
    """
        Model validation step:
         1. Re-rank a given run
         2. Save the re-ranked run
         3. Evaluate results

        :param model:           a model reference.
        :param train_params:    training parameters
        :param dataset:         validation dataset
        :param orig_run:        a run to re-rank
        :param qrelf:           QREL files
        :param run_filename:    a file name to store the *RE-RANKED* run
        :return: validation score


    """
    sync_out_streams()

    rerank_run = run_model(model, train_params, dataset, orig_run)
    eval_metric = train_params.eval_metric

    sync_out_streams()

    print(f'\n', f'Evaluating run with QREL file {qrelf} using metric {eval_metric}')

    sync_out_streams()

    # Let us always save the run
    return get_eval_results(use_external_eval=train_params.use_external_eval,
                          eval_metric=eval_metric,
                          rerank_run=rerank_run,
                          qrel_file=qrelf,
                          run_file=run_filename)


def run_model(model, train_params, dataset, orig_run, desc='valid'):
    auto_cast_class, scaler = get_amp_processors(train_params.amp)
    rerank_run = {}
    clean_memory(train_params.device_name)
    cand_score_weight = torch.FloatTensor([train_params.cand_score_weight]).to(train_params.device_name)
    with torch.no_grad(), \
            tqdm(total=sum(len(r) for r in orig_run.values()), ncols=80, desc=desc, leave=False,  file=TQDM_FILE) as pbar:

        model.eval()
        d = {}
        for records in data.iter_valid_records(model,
                                               train_params.device_name,
                                               dataset, orig_run,
                                               train_params.batch_size_val,
                                               train_params.max_query_len, train_params.max_doc_len):
            with auto_cast_class():
                scores = model(records[QUERY_TOK_FIELD],
                       records[QUERY_MASK_FIELD],
                       records[DOC_TOK_FIELD],
                       records[DOC_MASK_FIELD]) + records[CAND_SCORE_FIELD] * cand_score_weight

            for qid, did, score in zip(records[QUERY_ID_FIELD], records[DOC_ID_FIELD], scores):
                rerank_run.setdefault(qid, {})[did] = score.item()
            pbar.update(len(records[QUERY_ID_FIELD]))

    return rerank_run


def do_train(sync_barrier,
              device_qty, master_port, rank, is_master_proc,
              dataset,
              qrels, qrel_file_name,
              train_pairs, valid_run,
              valid_run_dir, valid_checkpoints,
              model_out_dir,
              model_holder, loss_obj, train_params):
    if device_qty > 1:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(master_port)
        dist.init_process_group(PYTORCH_DISTR_BACKEND, rank=rank, world_size=device_qty)

    device_name = train_params.device_name

    bert_param_keys = model_holder.model.bert_param_names()

    if is_master_proc:
        print('Training parameters:')
        print(train_params)
        print('BERT parameters:')
        print(bert_param_keys)
        print('Loss function:', loss_obj.name())

    print('Device name:', device_name)

    model_holder.model.to(device_name)

    lr = train_params.init_lr
    bert_lr = train_params.init_bert_lr
    epoch_lr_decay = train_params.epoch_lr_decay
    weight_decay = train_params.weight_decay
    momentum = train_params.momentum

    top_valid_score = None

    train_stat = {}

    validation_timer = ValidationTimer(valid_checkpoints)
    valid_scores_holder = dict()
    for epoch in range(train_params.epoch_qty):

        all_params = [(k, v) for k, v in model_holder.model.named_parameters() if v.requires_grad]
        # BERT parameters use a special learning weight
        bert_params =     {'params': [v for k, v in all_params if     k in bert_param_keys], 'lr': bert_lr}
        non_bert_params = {'params': [v for k, v in all_params if not k in bert_param_keys]}

        if train_params.optim == OPT_ADAMW:
            optimizer = torch.optim.AdamW([non_bert_params, bert_params],
                                       lr=lr, weight_decay=weight_decay)
        elif train_params.optim == OPT_SGD:
            optimizer = torch.optim.SGD([non_bert_params, bert_params],
                                         lr=lr, weight_decay=weight_decay,
                                         momentum=momentum)
        else:
            raise Exception('Unsupported optimizer: ' + train_params.optim)

        bpte = train_params.batches_per_train_epoch
        max_train_qty = data.train_item_qty_upper_bound(train_pairs)

        if bpte is not None and bpte >= 0:
            max_train_qty = min(max_train_qty, int(bpte) * train_params.batch_size)
            print(f'Setting the number of train instances to {max_train_qty} b/c batches_per_train_epoch={bpte}')

        start_train_time = time.time()

        if max_train_qty > 0:

            lr_steps = int(math.ceil(max_train_qty / train_params.batch_size))
            scheduler = None
            if train_params.warmup_pct:
                if is_master_proc:
                    print('Using a scheduler with a warm-up for %f steps' % train_params.warmup_pct)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                                total_steps=lr_steps,
                                                                max_lr=[lr, bert_lr],
                                                                anneal_strategy='linear',
                                                                pct_start=train_params.warmup_pct)
            if is_master_proc:
                print('Optimizer', optimizer)

            loss = train_iteration(model=model_holder.model, sync_barrier=sync_barrier,
                               is_master_proc=is_master_proc,
                               device_qty=device_qty, loss_obj=loss_obj,
                               train_params=train_params, max_train_qty=max_train_qty,
                               valid_run=valid_run, valid_qrel_filename=qrel_file_name,
                               optimizer=optimizer, scheduler=scheduler,
                               dataset=dataset, train_pairs=train_pairs, qrels=qrels,
                               validation_timer=validation_timer, valid_run_dir=valid_run_dir,
                               valid_scores_holder=valid_scores_holder,
                               save_last_snapshot_every_k_batch=train_params.save_last_snapshot_every_k_batch,
                               model_out_dir=model_out_dir)
        else:
            loss = 0

        end_train_time = time.time()

        if is_master_proc:

            if train_params.save_epoch_snapshots:
                print('Saving the model epoch snapshot')
                model_holder.save_all(os.path.join(model_out_dir, f'model.{epoch}'))

            os.makedirs(model_out_dir, exist_ok=True)

            print(f'train epoch={epoch} loss={loss:.3g} lr={lr:g} bert_lr={bert_lr:g}')

            sync_out_streams()

            start_val_time = time.time()

            # Run validation if the validation type is
            run_val = (train_params.valid_type == VALID_ALWAYS) or \
                      ((train_params.valid_type == VALID_LAST) and epoch + 1 == train_params.epoch_qty)

            if run_val:
                valid_score = validate(model_holder.model,
                                       train_params, dataset,
                                       valid_run,
                                       qrelf=qrel_file_name,
                                       run_filename=os.path.join(model_out_dir, f'{epoch}.run'))
            else:
                print(f'No validation at epoch: {epoch}')
                valid_score = None

            end_val_time = time.time()

            sync_out_streams()

            if valid_score is not None:
                print(f'validation epoch={epoch} score={valid_score:.4g}')

            train_stat[epoch] = {'loss' : loss,
                                  'score' : valid_score,
                                  'lr' : lr,
                                  'bert_lr' : bert_lr,
                                  'train_time' : end_train_time - start_train_time,
                                  'validation_time' : end_val_time - start_val_time}

            save_json(os.path.join(model_out_dir, 'train_stat.json'), train_stat)

            if run_val:
                if top_valid_score is None or valid_score > top_valid_score:
                    top_valid_score = valid_score
                    print('new top validation score, saving the whole model')
                    model_holder.save_all(os.path.join(model_out_dir, 'model.best'))
            else:
                print('Saving the whole model')
                model_holder.save_all(os.path.join(model_out_dir, 'model.best'))

        # We must sync here or else non-master processes would start training and they
        # would timeout on the model averaging barrier. However, the wait time here
        # can be much longer. This is actually quite lame, because validation
        # should instead be split accross GPUs, but validation is usually pretty quick
        # and this should work as a (semi-)temporary fix
        if device_qty > 1:
            try:
                sync_barrier.wait(BARRIER_WAIT_VALIDATION_TIMEOUT)
            except BrokenBarrierError:
                raise Exception('A model validation synchronization timeout!')

        lr *= epoch_lr_decay
        bert_lr *= epoch_lr_decay


def main_cli():
    parser = argparse.ArgumentParser('model training and validation')

    add_model_init_basic_args(parser, add_device_name=True, add_init_model_weights=True, mult_model=False)

    parser.add_argument('--max_query_len', metavar='max. query length',
                        type=int, default=data.DEFAULT_MAX_QUERY_LEN,
                        help='max. query length')

    parser.add_argument('--max_doc_len', metavar='max. document length',
                        type=int, default=data.DEFAULT_MAX_DOC_LEN,
                        help='max. document length')

    parser.add_argument('--datafiles', metavar='data files', help='data files: docs & queries',
                        type=argparse.FileType('rt'), nargs='+', required=True)

    parser.add_argument('--qrels', metavar='QREL file', help='QREL file',
                        type=argparse.FileType('rt'), required=True)

    parser.add_argument('--train_pairs', metavar='paired train data', help='paired train data',
                        type=argparse.FileType('rt'), required=True)

    parser.add_argument('--valid_run', metavar='validation file', help='validation file',
                        type=argparse.FileType('rt'), required=True)

    parser.add_argument('--model_out_dir',
                        metavar='model out dir', help='an output directory for the trained model',
                        required=True)

    parser.add_argument('--epoch_qty', metavar='# of epochs', help='# of epochs',
                        type=int, default=10)

    parser.add_argument('--no_cuda', action='store_true',
                        help='Use no CUDA')

    parser.add_argument('--valid_type',
                        default=VALID_ALWAYS,
                        choices=[VALID_ALWAYS, VALID_LAST, VALID_NONE],
                        help='validation type')

    parser.add_argument('--warmup_pct', metavar='warm-up fraction',
                        default=None, type=float,
                        help='use a warm-up/cool-down learning-reate schedule')

    parser.add_argument('--device_qty', type=int, metavar='# of device for multi-GPU training',
                        default=1, help='# of GPUs for multi-GPU training')

    parser.add_argument('--batch_sync_qty', metavar='# of batches before model sync',
                        type=int, default=4, help='model syncronization frequency for multi-GPU trainig in the # of batche')

    parser.add_argument('--master_port', type=int, metavar='pytorch master port',
                        default=None, help='pytorch master port for multi-GPU training')

    parser.add_argument('--print_grads', action='store_true',
                        help='print gradient norms of parameters')

    parser.add_argument('--save_epoch_snapshots', action='store_true',
                        help='save model after each epoch')

    parser.add_argument('--save_last_snapshot_every_k_batch',
                        metavar='debug: save latest snapshot every k batch',
                        type=int, default=None,
                        help='debug option: save latest snapshot every k batch')

    parser.add_argument('--seed', metavar='random seed', help='random seed',
                        type=int, default=42)

    parser.add_argument('--optim', metavar='optimizer', choices=[OPT_SGD, OPT_ADAMW], default=OPT_ADAMW,
                        help='Optimizer')

    parser.add_argument('--loss_margin', metavar='loss margin', help='Margin in the margin loss',
                        type=float, default=1.0)

    # If we use the listwise loss, it should be at least two negatives by default
    parser.add_argument('--neg_qty_per_query', metavar='listwise negatives',
                        help='Number of negatives per query for a listwise losse',
                        type=int, default=2)

    parser.add_argument('--init_lr', metavar='init learn. rate',
                        type=float, default=0.001, help='initial learning rate for BERT-unrelated parameters')

    parser.add_argument('--momentum', metavar='SGD momentum',
                        type=float, default=0.9, help='SGD momentum')

    parser.add_argument('--cand_score_weight', metavar='candidate provider score weight',
                        type=float, default=0.0,
                        help='a weight of the candidate generator score used to combine it with the model score.')

    parser.add_argument('--init_bert_lr', metavar='init BERT learn. rate',
                        type=float, default=0.00005, help='initial learning rate for BERT parameters')

    parser.add_argument('--epoch_lr_decay', metavar='epoch LR decay',
                        type=float, default=1.0, help='per-epoch learning rate decay')

    parser.add_argument('--weight_decay', metavar='weight decay',
                        type=float, default=0.0, help='optimizer weight decay')

    parser.add_argument('--batch_size', metavar='batch size',
                        type=int, default=32, help='batch size')

    parser.add_argument('--batch_size_val', metavar='val batch size',
                        type=int, default=32, help='validation batch size')

    parser.add_argument('--backprop_batch_size', metavar='backprop batch size',
                        type=int, default=1,
                        help='batch size for each backprop step')

    parser.add_argument('--batches_per_train_epoch', metavar='# of rand. batches per epoch',
                        type=int, default=None,
                        help='# of random batches per epoch: 0 tells to use all data')

    parser.add_argument('--max_query_val', metavar='max # of val queries',
                        type=int, default=0,
                        help='max # of validation queries: 0 tells to use all data')

    parser.add_argument('--no_shuffle_train', action='store_true',
                        help='disabling shuffling of training data')

    parser.add_argument('--use_external_eval', action='store_true',
                        help='use external eval tools: gdeval or trec_eval')

    parser.add_argument('--eval_metric', choices=METRIC_LIST, default=METRIC_LIST[0],
                        help='Metric list: ' +  ','.join(METRIC_LIST), 
                        metavar='eval metric')

    parser.add_argument('--loss_func', choices=LOSS_FUNC_LIST,
                        default=PairwiseSoftmaxLoss.name(),
                        help='Loss functions: ' + ','.join(LOSS_FUNC_LIST))

    parser.add_argument('--amp', action='store_true', help="Use automatic mixed-precision")

    parser.add_argument('--json_conf', metavar='JSON config',
                        type=str, default=None,
            help='a JSON config (simple-dictionary): keys are the same as args, takes precedence over command line args')

    parser.add_argument('--valid_run_dir', metavar='', type=str, default=None, help='directory to store predictions on validation set')
    parser.add_argument('--valid_checkpoints', metavar='', type=str, default=None, help='validation checkpoints (in # of batches)')

    args = parser.parse_args()

    all_arg_names = vars(args).keys()

    if args.json_conf is not None:
        conf_file = args.json_conf
        print(f'Reading configuration variables from {conf_file}')
        add_conf = read_json(conf_file)
        for arg_name, arg_val in add_conf.items():
            arg_name : str
            if arg_name not in all_arg_names and not arg_name.startswith(MODEL_PARAM_PREF):
                print(f'Invalid option in the configuration file: {arg_name}')
                sys.exit(1)
            arg_default = getattr(args, arg_name, None)
            exp_type = type(arg_default)
            if arg_default is not None and type(arg_val) != exp_type:
                print(f'Invalid type in the configuration file: {arg_name} expected type: '+str(type(exp_type)) + f' default {arg_default}')
                sys.exit(1)
            print(f'Using {arg_name} from the config')
            setattr(args, arg_name, arg_val)


    if args.save_last_snapshot_every_k_batch is not None and args.save_last_snapshot_every_k_batch < 2:
        print('--save_last_snapshot_every_k_batch should be > 1')
        sys.exit(1)

    print(args)
    sync_out_streams()

    set_all_seeds(args.seed)

    loss_name = args.loss_func
    if loss_name == PairwiseSoftmaxLoss.name():
        loss_obj = PairwiseSoftmaxLoss()
    elif loss_name == CrossEntropyLossWrapper.name():
        loss_obj = CrossEntropyLossWrapper()
    elif loss_name == MultiMarginRankingLossWrapper.name():
        loss_obj = MultiMarginRankingLossWrapper(margin = args.loss_margin)
    elif loss_name == PairwiseMarginRankingLossWrapper.name():
        loss_obj = PairwiseMarginRankingLossWrapper(margin = args.loss_margin)
    else:
        print('Unsupported loss: ' + loss_name)
        sys.exit(1)

    print('Loss:', loss_obj)

    # For details on our serialization approach, see comments in the ModelWrapper
    model_holder : ModelSerializer = None

    if args.init_model is not None:
        print('Loading a complete model from:', args.init_model.name)
        model_holder = ModelSerializer.load_all(args.init_model.name)
    else:
        if args.model_name is None:
            print('--model_name argument must be provided unless --init_model points to a fully serialized model!')
            sys.exit(1)
        if args.init_model_weights is not None:
            model_holder = ModelSerializer(args.model_name)
            model_holder.create_model_from_args(args)
            print('Loading model weights from:', args.init_model_weights.name)
            model_holder.load_weights(args.init_model_weights.name, strict=False)
        else:
            model_holder = ModelSerializer(args.model_name)
            print('Creating the model from scratch!')
            model_holder.create_model_from_args(args)

    if args.neg_qty_per_query < 1:
        print('A number of negatives per query cannot be < 1')
        sys.exit(1)

    os.makedirs(args.model_out_dir, exist_ok=True)
    print(model_holder.model)
    sync_out_streams()

    dataset = data.read_datafiles(args.datafiles)
    qrelf = args.qrels.name
    qrels = read_qrels_dict(qrelf)
    train_pairs_all = data.read_pairs_dict(args.train_pairs)
    valid_run = read_run_dict(args.valid_run.name)
    max_query_val = args.max_query_val
    query_ids = list(valid_run.keys())
    if max_query_val > 0:
        query_ids = query_ids[0:max_query_val]
        valid_run = {k: valid_run[k] for k in query_ids}

    print('# of eval. queries:', len(query_ids), ' in the file', args.valid_run.name)


    device_qty = args.device_qty
    master_port = args.master_port
    if device_qty > 1:
        if master_port is None:
            print('Specify a master port for distributed training!')
            sys.exit(1)

    processes = []

    is_distr_train = device_qty > 1

    qids = []

    if is_distr_train:
        qids = list(train_pairs_all.keys())

    sync_barrier = Barrier(device_qty)

    # We must go in the reverse direction, b/c
    # rank == 0 trainer is in the same process and
    # we call the function do_train in the same process,
    # i.e., this call is blocking processing and
    # prevents other processes from starting.
    for rank in range(device_qty - 1, -1, -1):
        if is_distr_train:
            device_name = f'cuda:{rank}'
        else:
            device_name = args.device_name
            if args.no_cuda:
                device_name = DEVICE_CPU

        # When we have only a single GPP, the main process is its own master
        is_master_proc = rank == 0

        train_params = TrainParams(init_lr=args.init_lr, init_bert_lr=args.init_bert_lr,
                                   momentum=args.momentum, amp=args.amp,
                                    warmup_pct=args.warmup_pct, batch_sync_qty=args.batch_sync_qty,
                                    epoch_lr_decay=args.epoch_lr_decay, weight_decay=args.weight_decay,
                                    backprop_batch_size=args.backprop_batch_size,
                                    batches_per_train_epoch=args.batches_per_train_epoch,
                                    save_epoch_snapshots=args.save_epoch_snapshots,
                                    save_last_snapshot_every_k_batch=args.save_last_snapshot_every_k_batch,
                                    batch_size=args.batch_size, batch_size_val=args.batch_size_val,
                                    # These lengths must come from the model serializer object, not from the arguments,
                                    # because they can be overridden when the model is loaded.
                                    max_query_len=model_holder.max_query_len, max_doc_len=model_holder.max_doc_len,
                                    epoch_qty=args.epoch_qty, device_name=device_name,
                                    cand_score_weight=args.cand_score_weight,
                                    neg_qty_per_query=args.neg_qty_per_query,
                                    use_external_eval=args.use_external_eval, eval_metric=args.eval_metric.lower(),
                                    print_grads=args.print_grads,
                                    shuffle_train=not args.no_shuffle_train,
                                    valid_type=args.valid_type,
                                    optim=args.optim)

        train_pair_qty = len(train_pairs_all)
        if is_distr_train or train_pair_qty < device_qty:
            tpart_qty = int((train_pair_qty + device_qty - 1) / device_qty)
            train_start = rank * tpart_qty
            train_end = min(train_start + tpart_qty, len(qids))
            train_pairs = { k : train_pairs_all[k] for k in qids[train_start : train_end] }
        else:
            train_pairs = train_pairs_all
        print('Process rank %d device %s using %d training pairs out of %d' %
              (rank, device_name, len(train_pairs), train_pair_qty))

        valid_checkpoints = [] if args.valid_checkpoints is None \
                            else list(map(int, args.valid_checkpoints.split(',')))
        param_dict = {
            'sync_barrier': sync_barrier,
            'device_qty' : device_qty, 'master_port' : master_port,
            'rank' : rank, 'is_master_proc' : is_master_proc,
            'dataset' : dataset,
            'qrels' : qrels, 'qrel_file_name' : qrelf,
            'train_pairs' : train_pairs,
            'valid_run' : valid_run,
            'valid_run_dir' : args.valid_run_dir,
            'valid_checkpoints' : valid_checkpoints,
            'model_out_dir' : args.model_out_dir,
            'model_holder' : model_holder, 'loss_obj' : loss_obj, 'train_params' : train_params
        }

        if is_distr_train and not is_master_proc:
            p = Process(target=do_train, kwargs=param_dict)
            p.start()
            processes.append(p)
        else:
            do_train(**param_dict)

    for p in processes:
        join_and_check_stat(p)

    if device_qty > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    # A light-weight subprocessing + this is a must for multi-processing with CUDA
    enable_spawn()
    main_cli()
