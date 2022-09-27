import sys
import time
from datetime import timedelta
import logging
import argparse
import random
import numpy as np
import torch

from .data import get_dataset_cls
from .model import get_model_cls
from .training import get_trainer_cls
from .predictor import get_predictor_cls

logger = logging.getLogger()


class LogFormatter():
    """
    Using facebook codes.
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
        # 注释下面, 让换行加上缩紧.
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
    _LOGGER_FORMAT = True


def parse_args(parser=None, arglist=None):
    """
    将dataset和model的args也加上去, 这样显示帮助的时候能够将对应的组显示出来.
    """
    assert arglist is None or isinstance(arglist, list), f'arglist should be a list like sys.argv[1:] or None.'
    assert parser is None or parser.add_help == False, f'在parser初始化的时候, 使用add_help=False, e.g. parser = argparse.ArgumentParser(add_help=False)'

    parser = parser or argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='store_true')
    parser.add_argument('--seed', type=int, default=3)
    
    # TODO: 添加全部的dataset和models等等, 默认参数, choices等等
    # dataset
    parser.add_argument('--dataset', type=str, default='translation_dataset')
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

    # help
    if args.help:
        print(parser.format_help())
        parser.exit()
    return parser.parse_args(arglist)


def set_random_state(seed):
    if not seed:
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
