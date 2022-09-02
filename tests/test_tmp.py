"""
临时测试用了, 不是测试multihead用的
"""




import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--data-dir', type=str)
args, _ = parser.parse_known_args()
if args.data_dir:
    parser.add_argument('--know', type=str)
else:
    parser.add_argument('--end', type=str)

# args = parser.parse_args()

print(args)

from torch.nn import Linear
from torch.nn import functional as F

F.multi_head_attention_forward