import sys
import time
from datetime import timedelta
import logging
import argparse
import random
import numpy as np
import torch
import json 

from lessdl.utils import bool_flag
from lessdl.data import get_dataset_cls
from lessdl.model import get_model_cls, get_arch_arch
from lessdl.training import get_trainer_cls, load_exp_args, load_args
from lessdl.training.callbacks import Checkpoint
from lessdl.predictor import get_predictor_cls

logger = logging.getLogger()


class LogFormatter():
    """Using LogFormatter from fairseq
    """
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        # Comment below to add index to each line
        # message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


_LOGGER_FORMAT = False
if not _LOGGER_FORMAT:
    logging.basicConfig(level=logging.INFO)
    logger.handlers = []
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(LogFormatter())
    logger.addHandler(console_handler)
    # logger.info('Setup logger')
    _LOGGER_FORMAT = True


def set_random_state(seed):
    if not seed:
        logger.info('No seed is set.')
        return 
    assert isinstance(seed, int)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.warning("Using seed will turn on the CUDNN deterministic setting, which can slow down your training considerably!")


def parse_args(parser=None, arglist=None):
    """Add trainer/model/dataset... args if they are specified
    """
    assert arglist is None or isinstance(arglist, list), f'arglist should be a list like sys.argv[1:] or None.'
    assert parser is None or parser.add_help == False, f'在parser初始化的时候, 使用add_help=False, e.g. parser = argparse.ArgumentParser(add_help=False)'

    parser = parser or argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='store_true')
    parser.add_argument('--seed', type=int, default=3)
    
    # TODO: 添加全部的dataset和models等等, 默认参数, choices等等
    # dataset
    parser.add_argument('--dataset', type=str, default='translation_dataset')
    parser.add_argument('--train-split', type=str, default='train', )
    parser.add_argument('--valid-split', type=str, default='valid', )
    parser.add_argument('--test-split', type=str, default='test', help='use none to disable it.')
    args, _ = parser.parse_known_args(arglist)
    # 如果没有help, 必须要指定dataset
    if args.dataset or not args.help:
        dataset_cls = get_dataset_cls(args.dataset)
        dataset_cls.add_args(parser, arglist)
    
    # model
    parser.add_argument('--model', type=str)
    parser.add_argument('--arch', type=str)
    args, _ = parser.parse_known_args(arglist)
    if args.model or args.arch:
        args, _ = parser.parse_known_args(arglist)
        model_cls = get_model_cls(args.model, args.arch)
        model_cls.add_args(parser, arglist)
        args, _ = parser.parse_known_args(arglist)

    # training and evaluation
    # loss is added in trainer
    parser.add_argument('--trainer', type=str, default='basictrainer')
    args, _ = parser.parse_known_args(arglist)
    if args.trainer:
        trainer_cls = get_trainer_cls(args.trainer)
        trainer_cls.add_args(parser, arglist)

    # predictor
    parser.add_argument('--predictor', type=str)
    args, _ = parser.parse_known_args(arglist)
    if args.predictor:
        predictor_cls = get_predictor_cls(args.predictor)
        predictor_cls.add_args(parser, arglist)

    # eval best checkpoint 
    parser.add_argument('--evaluate-best-ckpt', type=bool_flag, default=True, help='Whether to evalute best checkpoit using test_data after training')
    # restore from old exp_dir 
    parser.add_argument('--restore-exp-dir', type=str, default=None, help='Restore args from old exp_dir.')
    # parser.add_argument('--restore-mode', type=str, default='best', choices=['best', 'last'], help='Restore last or best ckpt')
    # parser.add_argument('--restore-exp-ckpt', type=str, default=None, help='Restore ckpt from ')
    parser.add_argument('--evaluate-only', type=bool_flag, default='False', help='Whether only evaluation and no training.')

    # show help
    if args.help:
        print(parser.format_help())
        parser.exit()
    
    return parser.parse_args(arglist)


def run_main(args, evaluate_best_ckpt=True, evaluate_only=False):
    #
    if args.restore_exp_dir:
        logger.info('Load exp args from {}'.format(args.restore_exp_dir))
        load_exp_args(args, args.restore_exp_dir, overwrite=True, argline=None)

    set_random_state(args.seed)

    dataset_cls = get_dataset_cls(args.dataset)
    _, trainer, model = build_from_args(args)
    logger.info(f'Model:\n{model}')
    logger.info(f'Args: {args}')
    if args.restore_exp_dir:
        # restore last ckpt 
        # logger.info('Restore checkpoint from {}'.format(args.restore_exp_dir))
        # Checkpoint(args.exp_dir).restore
        logger.warn('Not implemented ----------------------------------')

    if not evaluate_only:
        trainer.train()

    # evaluate
    if args.test_split.lower() == 'none':
        logger.warn('test_data is None and do not evaluate on best checkpoint')
        return
    test_data = dataset_cls.build(args, args.test_split)
    if evaluate_best_ckpt:
        logger.info('Reload best checkpoint and test on dataset ...')
        trainer.ckpt.restore(best=True)
    if evaluate_best_ckpt or evaluate_only:
        trainer.evaluate(test_data)


def build_from_args(args, need_dataset=True, need_trainer=True, need_model=True):
    """
    Return:
        ((train_data, val_data), trainer, model)
    """
    ds = (None, None)
    trainer = None
    model = None
    # dataset
    if need_dataset:
        dataset_cls = get_dataset_cls(args.dataset)
        train_data = dataset_cls.build(args, args.train_split)
        valid_data = dataset_cls.build(args, args.valid_split)
        ds = (train_data, valid_data)

    # model
    if need_model:
        model_cls = get_model_cls(args.model, args.arch)
        if args.arch:
            arch = get_arch_arch(args.arch)
            arch(args)
        model = model_cls.build(args, train_data)

    # build training related objects(optimizer,lr_scheduler,callbacks) and train
    if need_trainer:
        trainer_cls = get_trainer_cls(args.trainer)
        trainer = trainer_cls.build(args, model, train_data, valid_data)

    return ds, trainer, model


def build_from_exp(exp_dir, best=False, last=False, ckpt_path_prefix=''):
    parser = argparse.ArgumentParser(add_help=False)
    args, _ = parser.parse_known_args()
    load_exp_args(args, exp_dir, overwrite=True, argline=[])
    (train_data, val_data), trainer, model = build_from_args(args)
    ckpt_cb = Checkpoint(exp_dir, [])
    ckpt_cb.set_model(model)
    if best or last or ckpt_path_prefix:
        ckpt_cb.restore(best=best, last=last, prefix=ckpt_path_prefix, map_location=next(model.parameters()).device)
    return (train_data, val_data), trainer, model
