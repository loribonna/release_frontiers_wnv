import os
# SET CUDA REPRODUCIBLE
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
# COMPATIBILITY WITH OLDER VERSIONS
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import argparse
import copy
import random
from utils.unknown_args import create_params_list, merge_keys, parse_unknown_args, check_required_args, get_default_args

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import RandomSampler

import utils
from dataset.dataset_AM_graph import AM_Dataset as AM_Dataset_graph, id_collate as id_collate_graph
from dataset.dataset_AM_temporal import AM_Dataset as AM_Dataset_temporal, id_collate as id_collate_temporal

from dataset.graph_dataset_utils import DatasetUtils as GraphDatasetUtils
from dataset.temporal_dataset_utils import DatasetUtils as TemporalDatasetUtils

from utils.job_configs import set_params
from metrics.metric import *
from models.baseline import Resnet_baseline as Resnet_baseline_temporal
from models.ResnetGraph import Resnet_baseline as Resnet_baseline_graph
from test import test
from train import train
import utils.globals

def parse_args(parser):
    parser.add_argument('--mode', type=str, choices=['graph', 'temporal', 'base'], default='temporal')
    parser.add_argument('--enable_distributed', type=int, default=0, choices=[0, 1])

    parser.add_argument('--debug_mode_test', action="store_true",
                        help='Enable debug mode on test?')
    parser.add_argument('--debug_mode', action="store_true",
                        help='Enable debug mode?')
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA training?')

    parser.add_argument('--label_threshold', type=int, default=0,
                        help='Threshold for culex pipien label?')
    parser.add_argument('--eval_last_year', type=int, default=0, choices=[0, 1],
                        help='Force eval only on last year (single split)?')

    args, others = parser.parse_known_args()

    # if args.mode != 'graph':
    #     args.json_db_file = "dataset/culex_pipiens_parsed.json"
    # else:
    #     args.json_db_file = f"dataset/culex_pipiens_graph_parsed.json"
    if args.mode != 'graph':
        args.json_db_file = "dataset/multitemporal_base.json"
    else:
        args.json_db_file = f"dataset/graph_base.json"

    if args.json_db_file.endswith('_base.json'):
        print("WARNING: Using example dataset!")
        print("Will only run eval on a single example.")
        args.eval_only = True

    args.cuda = not args.disable_cuda and torch.cuda.is_available()

    # set defaults for extra args
    required_ok, err_args = check_required_args(parser, others, vars(args))
    if not required_ok:
        print('The following arguments have errors:')
        print("\n".join(err_args))
    def_args = get_default_args(parser)
    args = argparse.Namespace(**merge_keys(def_args, vars(args)))

    # parse extra args
    others = parse_unknown_args(
        " ".join(others), single_values=False, try_parse=True)
    extra_params_list = create_params_list(others)
    assert len(
        extra_params_list) <= 1, "Only one parameter combination is allowed (single job run ONLY)"
    if len(extra_params_list) == 0:
        extra_params_list = {}
    else:
        extra_params_list = extra_params_list[0]

    # READ JSON CONFIG FILE
    args.data_basepath = utils.get_data_dir()
    params = argparse.Namespace()

    # for change params related to job-id
    params = set_params(params, args.mode)

    # merge extra args
    params.__dict__ = merge_keys(merge_keys(vars(params), extra_params_list), vars(args))

    if params.num_workers != 0:
        params.num_workers = os.sched_getaffinity(0).__len__() if hasattr(os, 'sched_getaffinity') else 4
    print('Using N_WORKERS:', params.num_workers)

    if args.debug_mode:
        s = "WARNING: Debug mode enabled."
        print("-" * (len(s) + 4))
        print(f"= {s} =")
        print("-" * (len(s) + 4))

    return args, params


def init_worker(id_worker):
    worker_seed = 7 + id_worker
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def update_best(model, test_loader, params, args, metrics, roc_score, loss_fn, epoch, split, device, log_dir):
    print(
        "Starting test for {} epoch(s) {} split".format(epoch + 1, split))
    metrics_test = test(model=model, test_loader=test_loader, loss_fn=loss_fn,
                        device=device, params=params, metrics=metrics, roc_score=roc_score,
                        split=split, args=args, log_dir=log_dir)

    if metrics_test is not None:
        # save best model params based on avg_pr_micro score on validation set
        f1_score = metrics_test['f1']
        if f1_score > utils.globals.f1_score:
            utils.globals.f1_score = f1_score
            utils.globals.model = copy.deepcopy(model.state_dict())
            state = {'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optim_dict': utils.globals.optimizer.state_dict(),
                     'scheduler': utils.globals.scheduler.state_dict()}
            utils.save_checkpoint(state,
                                  checkpoint=log_dir,
                                  split=split)


def get_datasets(params, args, db_utils, split, eval_last_year=False):
    if args.mode != 'graph':
        paths_labels_train, paths_labels_test = db_utils.paths_and_labels(
            split=split)

        AM_db_train = AM_Dataset_temporal(args, path_label=paths_labels_train, bands=params.bands, train=True,
                                          num_multi_images=params.num_multi_images,
                                          temporal_buffer=params.temporal_buffer, eval_last_year=eval_last_year)
        AM_db_test = AM_Dataset_temporal(args, path_label=paths_labels_test, bands=params.bands,
                                         num_multi_images=params.num_multi_images, train=False,
                                         temporal_buffer=params.temporal_buffer, eval_last_year=eval_last_year)  # no augm on test!
    else:
        paths_labels_train, paths_labels_test = db_utils.paths_and_labels(
            split=split)

        AM_db_train = AM_Dataset_graph(args, path_label=paths_labels_train, bands=params.bands, train=True,
                                       eval_last_year=eval_last_year,
                                       temporal_buffer=params.temporal_buffer,
                                       random_seed=params.seed, random_temporal=0)
        AM_db_test = AM_Dataset_graph(args, path_label=paths_labels_test, bands=params.bands,
                                      eval_last_year=eval_last_year, train=False,
                                      temporal_buffer=params.temporal_buffer,
                                      random_seed=params.seed, random_temporal=0)  # no augm on test!

    return AM_db_train, AM_db_test


def get_model(params, args, enable_grad_retain=False):
    if args.mode != 'graph':
        return Resnet_baseline_temporal(in_channels=params.in_channels,
                                        drop_rate=params.drop_rate, num_multi_images=params.num_multi_images,
                                        colorization=params.colorization,
                                        use_conditional_bn=args.mode != 'temporal',
                                        enable_grad_retain=enable_grad_retain)
    else:
        return Resnet_baseline_graph(in_channels=params.in_channels,
                                     drop_rate=params.drop_rate, colorization=params.colorization,
                                     use_conditional_bn=args.mode != 'temporal',
                                     enable_grad_retain=enable_grad_retain)


def get_device() -> torch.device:
    """
    Returns the least used GPU device if available else MPS or CPU.
    """
    def _get_device():
        # get least used gpu by used memory
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            gpu_memory = []
            for i in range(torch.cuda.device_count()):
                gpu_memory.append(torch.cuda.memory_allocated(i))
            device = torch.device(f'cuda:{np.argmin(gpu_memory)}')
            print(f'Using device {device}')
            return device
        try:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                print("WARNING: MSP support is still experimental. Use at your own risk!")
                return torch.device("mps")
        except BaseException:
            print("WARNING: Something went wrong with MPS. Using CPU.")
        return torch.device("cpu")

    # Permanently store the chosen device
    if not hasattr(get_device, 'device'):
        get_device.device = _get_device()

    return get_device.device


def main():
    # Initialize globals and exit handler
    utils.globals.init()

    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()
    args, params = parse_args(parser)
    init_split, init_epoch = 0, 0

    device = get_device()
    args.jobid = os.environ.get('SLURM_JOBID', 'local')
    params.jobid = args.jobid

    log_dir = os.path.join("experiments", "run_debug_nowand", params.log_dir)

    print(f"Saving logs in: {log_dir}")

    # set the torch seed
    if params.seed is not None:
        torch.manual_seed(params.seed)
        np.random.seed(params.seed)
        random.seed(params.seed)
        torch.use_deterministic_algorithms(True)

    # create dir for log file
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create the input data pipeline
    data_path = utils.get_data_dir()

    # Define the dataset utils and the split
    if args.eval_last_year or hasattr(args, 'eval_only') and args.eval_only:
        n_splits = 1
    else:
        if not hasattr(args, 'eval_only'):
            args.eval_only = False
        n_splits = 5

    params.n_split = n_splits

    in_channels = sum(params.bands)
    if args.mode == 'graph':
        in_channels -= 1
    params.in_channels = in_channels

    if args.mode != 'graph':
        AM_db_utils = TemporalDatasetUtils(json_file=params.json_db_file, random_seed=params.seed,
                                           n_splits=n_splits)
    else:
        AM_db_utils = GraphDatasetUtils(json_file=params.json_db_file, random_seed=params.seed,
                                        n_splits=n_splits)

    # list to save the metrics
    utils.globals.metrics = []

    # split
    print("Bands: ", params.bands)

    # repeat n_split time
    for split in range(init_split, params.n_split):
        if split != init_split:
            init_epoch = 0

        # Dataset definition: Train and Test
        AM_db_train, AM_db_test = get_datasets(
            params, args, AM_db_utils, split, eval_last_year=args.eval_last_year == 1)

        if not args.eval_only:
            print(f"Dataset len train: {AM_db_train.data_len}.")
        print(f"Dataset len test: {AM_db_test.data_len}.")

        # sampler
        if not args.eval_only:
            train_sampler = RandomSampler(AM_db_train)
        test_sampler = RandomSampler(AM_db_test)

        # Loader
        if params.seed is not None:
            g = torch.Generator()
            g.manual_seed(params.seed)
            init_fn = init_worker
        else:
            init_fn = None
            g = None
        collate_fn = id_collate_graph if args.mode == 'graph' else id_collate_temporal
        if not args.eval_only:
            train_loader = torch.utils.data.DataLoader(AM_db_train, batch_size=params.batch_size, pin_memory=True,
                                                    sampler=train_sampler, num_workers=params.num_workers,
                                                    worker_init_fn=init_fn, collate_fn=collate_fn, generator=g)
        test_loader = torch.utils.data.DataLoader(AM_db_test, batch_size=params.batch_size,
                                                  sampler=test_sampler, num_workers=params.num_workers,
                                                  worker_init_fn=init_fn, collate_fn=collate_fn, generator=g)

        print("- done.")

        # MODEL definition
        model = get_model(params, args)

        # Eventually load the colorization checkpoint
        if params.colorization:
            colorization_model_path = os.path.join(data_path, 'colorization_resnet.pth.tar')
            checkpoint = torch.load(colorization_model_path)
            model.model = utils.load_checkpoint_fix_statedict(model.model, checkpoint['state_dict'], params, args)

        in_channels = sum(params.bands)
        if args.mode == 'graph':
            in_channels -= 1
        # Eventually set the weights of the first layer
        model.set_weights_conv1(in_channels=in_channels)
        print("Weights of the first layer set to", in_channels, "channels")

        # check number of gpus
        if torch.cuda.device_count() > 1 and args.enable_distributed == 0:
            model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

        model = model.to(device)

        # Loss
        loss_fn = nn.CrossEntropyLoss()

        # METRICS
        metrics = metrics_def
        roc_score = roc

        # OPTIMIZER
        utils.globals.optimizer = optim.SGD(
            model.parameters(), lr=params.lr, momentum=params.momentum)

        # SCHEDULER
        if params.scheduler_type == 'Step':
            utils.globals.scheduler = optim.lr_scheduler.StepLR(
                utils.globals.optimizer, step_size=params.scheduler_step, gamma=0.1)
        else:
            utils.globals.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                utils.globals.optimizer, milestones=params.scheduler_milestones, gamma=0.1)

        # start training
        print("Split {}".format(split))
        print(
            "Starting training for {} epoch(s)".format(params.num_epochs))

        # SAVE THE BEST MODEL STATE DICT
        utils.globals.model = copy.deepcopy(model.state_dict())
        # BEST F1
        utils.globals.f1_score = 0.

        if not args.eval_only:
            for epoch in range(init_epoch, params.num_epochs):
                # Training
                print("Starting training for {} epoch(s)".format(epoch + 1))
                print("Epoch {}/{}".format(epoch + 1, params.num_epochs))
                train(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=utils.globals.optimizer,
                    device=device, params=params, metrics=metrics, roc_score=roc_score, args=args, split=split)
                # scheduler step
                utils.globals.scheduler.step()
                # test (for the moment I don't use a validation set...)
                if epoch % params.test_step == 0:
                    update_best(model, test_loader, params, args, metrics,
                                roc_score, loss_fn, epoch, split, device, log_dir)

        print("Starting final test...")
        model.load_state_dict(utils.globals.model)
        metric_test_best = test(model=model, test_loader=test_loader, loss_fn=loss_fn,
                                device=device, params=params, metrics=metrics, roc_score=roc_score,
                                split=split, args=args, log_dir=log_dir)
        if metric_test_best is None:
            exit(0)

        utils.globals.metrics.append(metric_test_best)
    
    print("METRICS")
    print(
        "Accuracy: {:05.3f}".format(sum([metric['accuracy'] for metric in utils.globals.metrics]) / float(params.n_split)))
    print(
        "Precision: {:05.3f}".format(sum([metric['precision'] for metric in utils.globals.metrics]) / float(params.n_split)))
    print(
        "Recall: {:05.3f}".format(sum([metric['recall'] for metric in utils.globals.metrics]) / float(params.n_split)))
    print(
        "f1: {:05.3f}".format(sum([metric['f1'] for metric in utils.globals.metrics]) / float(params.n_split)))
    print(
        "AUPRC: {:05.3f}".format(sum([metric['AUPRC'] for metric in utils.globals.metrics]) / float(params.n_split)))
    print(
        "Sensitivity: {:05.3f}".format(sum([metric['sensitivity'] for metric in utils.globals.metrics]) / float(params.n_split)))
    print(
        "Specificity: {:05.3f}".format(sum([metric['specificity'] for metric in utils.globals.metrics]) / float(params.n_split)))
    print(
        "TP: {:05.3f}".format(sum([metric['tp'] for metric in utils.globals.metrics]) / float(params.n_split)))
    print(
        "TN: {:05.3f}".format(sum([metric['tn'] for metric in utils.globals.metrics]) / float(params.n_split)))
    print(
        "FP: {:05.3f}".format(sum([metric['fp'] for metric in utils.globals.metrics]) / float(params.n_split)))
    print(
        "FN: {:05.3f}".format(sum([metric['fn'] for metric in utils.globals.metrics]) / float(params.n_split)))


if __name__ == '__main__':
    main()
