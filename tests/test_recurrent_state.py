
from tests.utils import get_testdata_trainer, replace_dropout, assert_tensor_eq
replace_dropout(nodrop=False)
import torch

from simdltk.module.transformer import generate_square_subsequent_mask
from simdltk.module.embedding import PositionalEmbedding



### 测试multi head attn 在不加prev_state的情况下, forward的是否正确.
### 测试multi head attn循环的情况下, 和不循环直接运行是否一致

### TODO delete
from tests.utils import multi_head_attention_forward
from torch.nn import functional as F
# F.multi_head_attention_forward = multi_head_attention_forward


trainer = get_testdata_trainer()
args = trainer.args
model = trainer.model
ma = model.decoder.layers[0].self_attn

for batch in trainer.train_loader:
    break

src = batch['src']
tgt = batch['tgt']
print('tgt_len', batch['tgt_len'])
# src = src[:, :15]
# tgt = tgt[:, :15]
padding_idx = trainer.train_dataset.src_vocab.pad()
emb = PositionalEmbedding(10000, trainer.args.encoder_embed_dim, padding_idx=padding_idx)
src_emb = emb(src).transpose(0, 1)
tgt_emb = emb(tgt).transpose(0, 1)

"""
先对比 self attn的形式
"""

def check_ma_recurent(static_kv=False):
    if static_kv:
        query, key, value = tgt_emb, src_emb, src_emb
    else:
        query, key, value = src_emb, src_emb, src_emb
    src_padding_mask = src.eq(padding_idx)
    src_self_attn_mask = generate_square_subsequent_mask(src.size(1), src.device)
    if not static_kv:
        out1, out1s = ma.forward(src_emb, src_emb, src_emb, 
            key_padding_mask=src_padding_mask, attn_mask=src_self_attn_mask, static_kv=static_kv
        )
    else:
        out1, out1s = ma.forward(tgt_emb, src_emb, src_emb, key_padding_mask=src_padding_mask, static_kv=static_kv,
            attn_mask=None,
        )
    print('batch size', args.batch_size, 'seq len', src.size(1))
    print('out1.shape', out1.shape)
    out2 = []
    prev_state = {}
    time = src.size(1) if not static_kv else tgt.size(1)
    for t in range(time):
        if not static_kv:
            t_emb = src_emb[t:t+1]
            t_padding_mask = src_padding_mask[:, t:t+1]
            tk = t_emb
            tv = t_emb
            t_self_attn_mask = src_self_attn_mask[t:t+1, :t+1]
        else:
            t_emb = tgt_emb[t:t+1]
            tk = src_emb
            tv = src_emb
            t_padding_mask = src_padding_mask
            t_self_attn_mask = None
        tout, tweights, prev_state = ma.forward(t_emb, tk, tv, key_padding_mask=t_padding_mask, attn_mask=t_self_attn_mask,
            prev_state=prev_state)
        out2.append(tout)
        print('t', t)
    out2 = torch.cat(out2, dim=0)
    print('out2.shape', out2.shape)
    # assert_tensor_eq(out1s['linv'][0], prev_state['linv'][0])
    # for lbl in ['link', 'linv']:
    #     print(lbl)
    #     print(type(out1s), type(prev_state))
    #     if not static_kv:
    #         assert_tensor_eq(out1s[lbl], prev_state[lbl], thr=1, max_thr=0.00001)

    assert_tensor_eq(out1, out2, thr=1, max_thr=0.00001)

print('check self attn')
check_ma_recurent(static_kv=False)
ma.eval()
check_ma_recurent(static_kv=False)
"""
再对比static kv的形式
"""
print('check static kv')
ma.train()
check_ma_recurent(static_kv=True)
ma.eval()
check_ma_recurent(static_kv=True)
ma.train()


### 测试transformer在循环和不循环的情况下是否输出一致
## 测试的时候, training=True,False都要测试
tgt_key_padding_mask = tgt.eq(padding_idx)  # B x L


def check_transformer_recurrent():
    model = trainer.model
    out1 = model.forward(src, tgt)
    print(list(out1.keys()))
    out2 = []
    prev_state = {}
    for t in range(tgt.size(1)):
        tin = tgt[:, t:t+1]
        o, prev_state = model.forward(src, tin, prev_state)
        out2.append(o)
    out2_logits = torch.cat([t['logits'] for t in out2], dim=1)
    padding_mask = tgt_key_padding_mask.unsqueeze(2).repeat(1, 1, out2_logits.size(-1))
    # 去掉pad的部分, 两者由于循环的原因, 导致不一样.
    out2_logits.masked_fill_(padding_mask, 0)
    memory = out2[-1]['memory']
    assert_tensor_eq(out1['memory'], memory)
    out1_logits = out1['logits']
    out1_logits.masked_fill_(padding_mask, 0)
    # assert_tensor_eq(out1['logits'][:, :2], out2_logits[:, :2], thr=0.001, max_thr=0.00001)
    # print((out1['logits'][:, 2] - out2_logits[:, 2]).abs() > 0.01)
    assert_tensor_eq(out1_logits, out2_logits, thr=0.1, max_thr=0.00001)


trainer.model.train()
check_transformer_recurrent()
trainer.model.eval()
check_transformer_recurrent()
## test_transformer也要修改测试, 修改成函数, 加上training=True,False的情况, 修改dropout在是否training的不同形式.


# TODO: 注释 prev_state有的时候, 将每个参数的每个时刻的shape都是加上去.
# F.multi_head_attn 的static kv的含义, 写上blog


print('OK!')


