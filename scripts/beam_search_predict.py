import os
import argparse
import torch

from lessdl import parse_args, logger, set_random_state
from lessdl.predictor import get_predictor_cls
from lessdl.training.utils import load_args, move_to_device
from lessdl.data import get_dataset_cls
from lessdl.data.dataloader import DataLoader
from lessdl.model import get_model_cls, get_arch_arch


"""
加载之前存储的args
构建模型和数据
构建predictor
进行预测, 输出预测的结果文件
"""


def index_to_text(tensor, vocab, skip_bos=True, skip_eos=True):
    assert isinstance(tensor, torch.Tensor)
    if len(tensor.shape) == 1:
        text = []
        for i, idx in enumerate(tensor):
            if i == 0 and skip_bos and idx == vocab.bos():
                continue
            # if idx == vocab.pad():
            #     break
            if skip_eos and idx == vocab.eos():
                break
            text.append(vocab.index_to_word(idx.item()))
        return text

    return [index_to_text(t, vocab, skip_bos, skip_eos) for t in tensor]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)

    args = parse_args(parser=parser)
    set_random_state(args.seed)
    
    if not hasattr(args, 'predictor') or args.predictor is None:
        parser.set_defaults(predictor='beam_search')
        predictor_cls = get_predictor_cls('beam_search')
        predictor_cls.add_args(parser)        
    args = parser.parse_args()
    assert hasattr(args, 'exp_dir'), 'exp_dir should be specified.'
    arg_file = args.exp_dir + '/kwargs.json'
    if os.path.exists(arg_file):
        logger.info(f'Loading args from {arg_file}')
        load_args(args, arg_file, overwrite=False)
    else:
        logger.info('No args file found.')
    print('Args', args)
    
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    dataset_cls = get_dataset_cls(args.dataset)
    test_data = dataset_cls.build(args, 'test')
    if hasattr(args, 'arch'):
        arch = get_arch_arch(args.arch)
        arch(args)
    model_cls = get_model_cls(args.model, args.arch)
    model = model_cls.build(args, test_data)
    if device:
        model.to(device)
    model_ckpt = os.path.join(args.exp_dir, 'best.pt')
    logger.info(f'Restore best model ckpt at {model_ckpt}')
    model.load_state_dict(torch.load(model_ckpt))
    print('Model', model)
    predictor_cls = get_predictor_cls(args.predictor)
    predictor = predictor_cls.build(args, test_data, model)
    
    batch_size = getattr(args, 'batch_size', None)
    max_batch_tokens = getattr(args, 'max_batch_tokens', None)
    assert batch_size or max_batch_tokens, 'Either batch_size or max_batch_tokens should be specified.'
    logger.info(f'Batch size {batch_size}')
    dl = DataLoader(test_data, batch_size=batch_size, max_batch_tokens=max_batch_tokens, shuffle=False,
        max_samples_in_memory=args.max_samples_in_memory, sort_key=getattr(args, 'sort_key', None),
    )
    
    for batch in dl:
        # move to device
        batch = move_to_device(batch, device)
        out = predictor.predict(batch)
        id = batch['_id']
        src_raw = batch['_src_raw']
        target_raw = batch['_target_raw']
        beam_topk = out['beam_topk']
        beam_topk = index_to_text(beam_topk, test_data.tgt_vocab, skip_eos=True, skip_bos=True)
        for ith, sr, tg, topk in zip(id, src_raw, target_raw, beam_topk):
            print(f'id-{ith}-src\t' + ' '.join(sr))
            print(f'id-{ith}-target\t' + ' '.join(tg))
            for j, k in enumerate(topk):
                print(f'id-{ith}-beam-{j}\t' + ' '.join(k))
    
    logger.info('Finished.')

