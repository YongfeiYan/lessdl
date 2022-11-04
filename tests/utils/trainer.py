

from lessdl import parse_args
from lessdl.model import get_model_cls, get_arch_arch
from lessdl.data import get_dataset_cls
from lessdl.training import get_trainer_cls
from lessdl import logger, parse_args, set_random_state


def build_trainer(argline):
    argline = argline.split()
    args = parse_args(arglist=argline)
    # dataset
    dataset_cls = get_dataset_cls(args.dataset)
    train_data = dataset_cls.build(args, 'train')
    valid_data = dataset_cls.build(args, 'valid')
    # model
    model_cls = get_model_cls(args.model, args.arch)
    if args.arch:
        arch = get_arch_arch(args.arch)
        arch(args)
    model = model_cls.build(args, train_data)
    # training
    trainer_cls = get_trainer_cls(args.trainer)
    trainer = trainer_cls.build(args, model, train_data, valid_data)
    
    return trainer, train_data, valid_data


_testdata_argline = """
    --exp-dir {}
    --dataset translation_dataset --src-language en --tgt-language de --data-dir tests/data/mt-en-de
    --arch transformer_iwslt_de_en --dropout 0.3
    --max-batch-tokens 1000 --num-workers 0 --max-samples-in-memory 1000 --epochs 3
    --optimizer adam,lr=5e-4,weight_decay=0.0001
    --lr-scheduler inverse_sqrt,warmup_updates=2,warmup_end_lr=5e-4
    --loss label_smoothed_cross_entropy --label-smoothing 0.1
"""

