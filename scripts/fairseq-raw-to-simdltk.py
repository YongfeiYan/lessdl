
"""
Convert fairseq raw dataset into simdltk readable dataset
Usage: 
    python fairseq-raw-to-simdltk.py raw_fairseq_dir save_dir
"""

import os
import shutil
import argparse


def trans_dict(src_dict, dst_vocab):
    """
    Convert dictories of different languages
    """
    # 15 <pad> <unk> <bos> <eos>
    # <pad> 0
    # <unk> 0
    # <bos> 0
    # <eos> 0
    with open(src_dict) as f:
        lines = []
        for line in f:
            line = line.rstrip('\n')
            if line:
                lines.append(line)
    with open(dst_vocab, 'w') as wt:
        print(f'{len(lines) + 4} <pad> <unk> <bos> <eos>\n<pad> 0\n<unk> 0\n<bos> 0\n<eos> 0', file=wt)
        print('\n'.join(lines), file=wt)


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str, help='source fariseq raw dataset dir')
    parser.add_argument('dst', type=str, help='saving dir')
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    # trans_dict(args.src + '/dict.de.txt', args.dst + '/vocab.de')
    for lbl in ['train', 'valid', 'test']:
        for lang in ['en', 'de']:
            print(lbl, lang)
            trans_dict(args.src + f'/dict.{lang}.txt', args.dst + f'/vocab.{lang}')
            shutil.copy(args.src + f'/{lbl}.en-de.{lang}', args.dst + f'/{lbl}.{lang}')

