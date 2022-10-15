import argparse

from lessdl.data import get_dataset_cls
from lessdl.model import get_model_cls, get_arch_arch
from lessdl.training import get_trainer_cls
from lessdl import logger, parse_args, set_random_state


def main(args, evaluate_best_ckpt=True):
    set_random_state(args.seed)
    # dataset
    dataset_cls = get_dataset_cls(args.dataset)
    train_data = dataset_cls.build(args, 'train')
    valid_data = dataset_cls.build(args, 'valid')
    test_data = dataset_cls.build(args, 'test')

    # model
    model_cls = get_model_cls(args.model, args.arch)
    if args.arch:
        arch = get_arch_arch(args.arch)
        arch(args)
    model = model_cls.build(args, train_data)
    logger.info(f'Model:\n{model}')

    # training
    trainer_cls = get_trainer_cls(args.trainer)
    trainer = trainer_cls.build(args, model, train_data, valid_data)
    logger.info(f'Args: {args}')
    trainer.train()
    if evaluate_best_ckpt:
        logger.info('Reload best checkpoint and test on dataset ...')
        trainer.ckpt.restore(best=True)
        trainer.evaluate(test_data)


if __name__ == '__main__':
    args = parse_args()
    main(args)


"""
测试运行脚本示例

lr=5e-4
python -u scripts/train.py \
    --exp-dir /tmp/train \
    --dataset translation_dataset --src-language en --tgt-language de --data-dir tests/data/mt-en-de \
    --arch transformer_iwslt_de_en \
    --batch-size 3 --num-workers 0 --max-samples-in-memory 10 --epochs 3 \
    --eval-every-n-epochs 1 --log-every-n-epochs 1 --grad-norm 0 \
    --optimizer adam,lr=$lr,weight_decay=0.0001 \
    --lr-scheduler inverse_sqrt,warmup_updates=4000,warmup_end_lr=$lr \
    --loss label_smoothed_cross_entropy --label-smoothing 0.1

"""


