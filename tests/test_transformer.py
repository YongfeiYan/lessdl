"""
将transformer和fairseq的transformer进行比较, 保证初始化和运行输出一致, 确保实现的正确性.
fairseq的版本为0.9.0
"""

import torch
from torch.nn import init
import argparse
from itertools import zip_longest
import copy


## 记录初始化, custom dropout, 方便对比.
init_func_back = {}
init_func_values = {}

def wrap_init_func(func, name):
    init_func_back[name] = func
    def wrapper(tensor, *args, **kwargs):
        print(name, 'called', 'tensor', type(tensor), 'id', id(tensor), 'shape', tensor.shape, 'args', args, 'kwargs', kwargs)
        key = f'{name} - {args} - {kwargs} - {tensor.shape}'
        # if isinstance(tensor, nn.Parameter):
        #     tensor = tensor.data
        assert isinstance(tensor, torch.Tensor), tensor
        # assert isinstance(tensor, nn.Parameter)
        if key not in init_func_values:
            print('set new value')
            value = func(tensor, *args, **kwargs)
            init_func_values[key] = value
        else:
            value = init_func_values[key]
            tensor.data = value.data.clone()
        tensor.initmethod = name
        tensor.initargs = key
        print('id', id(tensor), tensor.initmethod)
        return value
    return wrapper


init.xavier_normal_ = wrap_init_func(init.xavier_normal_, 'xavier_normal_')
init.xavier_uniform_ = wrap_init_func(init.xavier_uniform_, 'xavier_uniform_')
init.kaiming_normal_ = wrap_init_func(init.kaiming_normal_, 'kaiming_normal_')
init.kaiming_uniform_ = wrap_init_func(init.kaiming_uniform_, 'kaiming_uniform_')
init.constant_ = wrap_init_func(init.constant_, 'constant_')
init.normal_ = wrap_init_func(init.normal_, 'normal_')
init.uniform_ = wrap_init_func(init.uniform_, 'uniform_')
init.ones_ = wrap_init_func(init.ones_, 'ones_')
init.zeros_ = wrap_init_func(init.zeros_, 'zeros_')


# 修改parameter的copy函数, 也拷贝initmethod和initargs
def deepcp(self, memo):
    # print('deepcp', self, 'memo', memo)
    print('deecp id(self)', id(self), 'shape', self.data.shape)
    if id(self) in memo:
        return memo[id(self)]
    else:
        result = type(self)(self.data.clone(), self.requires_grad)
        self.initmethod
        result.initmethod = self.initmethod
        result.initargs = self.initargs
        memo[id(self)] = result
        print('result id', id(result))
        return result


from torch import nn
from torch.nn import functional as F
nn.Parameter.__deepcopy__ = deepcp


def drop(input, p, training=True, inplace=True):
    if not training:
        return input
    assert len(input.shape) == 3
    if len(input.shape) == 3:
        input[:, :, ::2] = p
    return input
# F.dropout(input, self.p, self.training, self.inplace)
F.dropout = drop


## 测试multiheadattn的参数是否都正确初始化了
from simdltk.module.activation import MultiheadAttention
from simdltk.module.transformer import TransformerEncoderLayer
def testinit(module):
    print(module, 'id', id(module))
    for p in module.parameters():
        assert hasattr(p, 'initmethod'), (p.shape, 'id', id(p))
    module = copy.deepcopy(module)
    for p in module.parameters():
        assert hasattr(p, 'initmethod'), (p.shape, 'id', id(p))

mha = MultiheadAttention(512, 4, 0)
print('mha')
testinit(mha)
from torch.nn import Linear
lin = Linear(3, 5)
print('linear')
testinit(lin)
enlayer = TransformerEncoderLayer(512, 4, 1024)
print('enlayer')
testinit(enlayer.linear1)
testinit(enlayer.linear2)
testinit(enlayer.self_attn)
testinit(enlayer)


## 构建我改写的transformer模型
from simdltk.data.dataset import TranslationDataset
from simdltk.model.transformer import Transformer

parser = argparse.ArgumentParser()
Transformer.add_args(parser)
args = parser.parse_args()
print('Args:', args)
"""dataset args"""
fileen = 'tests/data/en.txt'
filede = 'tests/data/de.txt'
vocaben = 'tests/data/vocab.en'
vocabde = 'tests/data/vocab.de'
args.src_name = 'src'
args.src_vocab_path = vocaben
args.src_file = fileen
args.src_max_sent_size = 250
args.tgt_name = 'tgt'
args.tgt_file = filede
args.tgt_vocab_path = vocabde
args.tgt_max_sent_size = 250
# dsende = PairIterDataset.build(args)
dsende = TranslationDataset('tests/data/mt-en-de', 'train', 'en', 'de')
"""model args"""
args.arch = 'transformer_iwslt_de_en'
args.model = None

# from simdltk.task import get_task_cls
from simdltk.model import get_arch_arch, get_model_cls
arch = get_arch_arch(args.arch)
arch(args)
print('transformer models')
model_cls = get_model_cls(args.model, args.arch)
model = model_cls.build(args, dsende)
for p in model.parameters():
    assert hasattr(p, 'initmethod'), p.shape
    assert hasattr(p, 'initargs'), p.shape


def assert_tensor_init_eq(a, b, check_args=True, check_value=True):
    """
    初始化方法相同而且相等
    """
    assert isinstance(a, torch.Tensor), type(a)
    assert isinstance(b, torch.Tensor), type(b)
    a.initmethod
    b.initmethod
    if check_args:
        a.initargs
        b.initargs
    assert a.initmethod == b.initmethod, (a.initmethod, b.initmethod)
    if check_args:
        assert a.initargs == b.initargs, (a.initargs, b.initargs)
    if check_value:
        assert (a - b).abs().sum().item() == 0, (a.initmethod, b.initmethod, a.initargs, b.initargs, a.shape, b.shape, 'ida', id(a), 'idb', id(b), (a - b).abs().sum().item(), '\na', a, '\nb', b)


def assert_layernom_init_eq(a: [nn.LayerNorm], b: [nn.LayerNorm]):
    if a is None and b is None:
        return 
    assert_tensor_init_eq(a.weight, b.weight)
    assert_tensor_init_eq(a.bias, b.bias)


def assert_dropout_init_eq(a, b):
    if a is None and b is None:
        return
    assert a.p == b.p, (a.p, b.p)
    # assert a.inplace == b.inplace, (a.inplace, b.inplace)


def assert_tensor_eq(a, b, thr=0.0):
    if isinstance(a, torch.BoolTensor):
        assert isinstance(b, torch.BoolTensor)
        print('change to int tensor')
        a = a.int()
        b = b.int()
    assert isinstance(a, torch.Tensor), type(a)
    assert isinstance(b, torch.Tensor), type(b)
    assert (a - b).abs().sum().item() <= thr, ((a - b).abs().sum().item(), a, b)


assert_tensor_eq(model.encoder.layers[0].self_attn.out_proj.weight, model.encoder.layers[1].self_attn.out_proj.weight)


## 构建fairseq的transformer
def build_fairseq_transformer():
    from fairseq.models.transformer import TransformerModel, transformer_iwslt_de_en
    parser = argparse.ArgumentParser()
    TransformerModel.add_args(parser)
    args = parser.parse_args()
    # 删除默认的参数, 用于architecture设置一些参数
    attrs = [k for k in args.__dir__() if not k.startswith('__')]
    for k in attrs:
        if getattr(args, k) is None:
            delattr(args, k)
    transformer_iwslt_de_en(args)
    ori_args = parser.parse_args()
    # 恢复默认的参数
    for k in ori_args.__dir__():
        if not hasattr(args, k):
            setattr(args, k, getattr(ori_args, k))
    class T:
        pass
    task = T()
    args.activation_dropout = args.dropout  # 不用默认的0
    print('args.droput', args.dropout)
    task.source_dictionary = dsende.src_vocab
    task.target_dictionary = dsende.tgt_vocab
    model = TransformerModel.build_model(args, task)
    return model

fmodel = build_fairseq_transformer()
print('fairseq model')
print(fmodel)


model.train()
fmodel.train()



def compare_embedding(model, fairseq_model):
    print('check encoder embedding')
    assert_tensor_init_eq(model.encoder_embed_tokens.weight, fairseq_model.encoder.embed_tokens.weight)
    # assert_tensor_init_eq(model.encoder_embed_positions.weight, fairseq_model.encoder.embed_positions.weight)
    print('check encoder embedding scale')
    assert model.encoder_embed_scale == fairseq_model.encoder.embed_scale
    print('check encoder layernorm')
    assert_layernom_init_eq(model.encoder_layernorm_embedding, fairseq_model.encoder.layernorm_embedding)
    print('check encoder embedding dropout')
    assert_dropout_init_eq(model.encoder_embed_dropout, fairseq_model.encoder.dropout_module)
    assert fairseq_model.encoder.encoder_layerdrop == 0, fairseq_model.encoder.encoder_layerdrop
    print('check encoder embedding lookup')
    mx, me = model.forward_embedding(batch['src'], model.encoder_embed_tokens, model.encoder_embed_positions,
        model.encoder_embed_dropout, model.encoder_embed_scale, model.encoder_layernorm_embedding)
    fx, fe = fairseq_model.encoder.forward_embedding(batch['src'])
    assert_tensor_eq(mx, fx)
    assert_tensor_eq(me, fe)

    print('check decoder embedding')
    assert_tensor_init_eq(model.decoder_embed_tokens.weight, fairseq_model.decoder.embed_tokens.weight)
    print('check decoder embedding scale')
    assert model.decoder_embed_scale == fairseq_model.decoder.embed_scale
    print('check decoder layernorm')
    assert_layernom_init_eq(model.decoder_layernorm_embedding, fairseq_model.decoder.layernorm_embedding)
    print('check decoder embedding dropout')
    assert_dropout_init_eq(model.decoder_embed_dropout, fairseq_model.decoder.dropout_module)
    assert fairseq_model.decoder.decoder_layerdrop == 0, fairseq_model.decoder.decoder_layerdrop
    # print('check decoder embedding lookup')
    assert fairseq_model.decoder.project_in_dim is None
    # mx, me = model.forward_embedding(batch['src'], model.decoder_embed_tokens, model.decoder_embed_positions,
    #     model.decoder_embed_dropout, model.decoder_embed_scale, model.decoder_layernorm_embedding)
    # fx, fe = fairseq_model.decoder.forward_embedding(batch['src'])
    # assert_tensor_eq(mx, fx)
    # assert_tensor_eq(me, fe)


def compare_multiheadatt(ah, bh):
    """
    比较的同时, 将参数初始化到相同, 便于以后检查输出.
    """
    assert ah.embed_dim == bh.embed_dim, (ah.embed_dim, bh.embed_dim)
    assert ah.num_heads == bh.num_heads, (ah.num_heads, bh.num_heads)
    assert ah.add_zero_attn == bh.add_zero_attn, (ah.add_zero_attn, bh.add_zero_attn)
    assert ah.dropout == bh.dropout_module.p, (ah.dropout, bh.dropout_module.p)
    # in_proj, 并且把fairseq self attn的初始化的值赋给自己的self-attn
    print(ah.in_proj_weight.shape, 'id', id(ah.in_proj_weight))
    print(ah.in_proj_weight.initmethod)
    assert_tensor_init_eq(ah.in_proj_weight, bh.k_proj.weight, check_args=False, check_value=False)
    # assert_tensor_init_eq(ah.in_proj_bias, bh.k_proj.bias, check_args=False, check_value=False)  # 和fairseq不一致, 一个是0, 一个是Linear自带的uniform
    # ah.in_proj_bias.data = torch.cat([bh.q_proj.bias.data, bh.k_proj.bias.data, bh.v_proj.bias.data], dim=0)
    ah.in_proj_weight.data = torch.cat([bh.q_proj.weight.data, bh.k_proj.weight.data, bh.v_proj.weight.data], dim=0)
    ah.in_proj_bias.data = torch.cat([bh.q_proj.bias.data, bh.k_proj.bias.data, bh.v_proj.bias.data])
    # out_proj
    assert_tensor_init_eq(ah.out_proj.weight, bh.out_proj.weight)
    assert_tensor_init_eq(ah.out_proj.bias, ah.out_proj.bias)


def compare_encoder(model, fairseq_model):
    # 对每个layer进行比较.
    print('check encoder')
    print('fairseq_model.encoder.dropout_module', fairseq_model.encoder.dropout_module.p)
    assert model.encoder.norm is None and fairseq_model.encoder.layer_norm is None, (model.endoer.norm, fairseq_model.encoder.layer_norm)
    for i, (l1, l2) in enumerate(zip_longest(model.encoder.layers, fairseq_model.encoder.layers)):
        # compare encoderlayer 
        print('check', i, 'th layer of encoder')
        assert l2.normalize_before is False
        # self attn dropout and layernorm
        compare_multiheadatt(l1.self_attn, l2.self_attn)
        assert_dropout_init_eq(l1.dropout1, l2.dropout_module)
        assert_layernom_init_eq(l1.norm1, l2.self_attn_layer_norm)
        # fc1 fc2 dropout layernorm
        assert_tensor_init_eq(l1.linear1.weight, l2.fc1.weight)
        assert_tensor_init_eq(l1.linear1.bias, l2.fc1.bias)
        assert_tensor_init_eq(l1.linear2.weight, l2.fc2.weight)
        assert_tensor_init_eq(l1.linear2.bias, l2.fc2.bias)
        assert_dropout_init_eq(l1.dropout, l2.activation_dropout_module)
        assert_dropout_init_eq(l1.dropout2, l2.dropout_module)
        assert_layernom_init_eq(l1.norm2, l2.final_layer_norm)
        assert F.relu == l2.activation_fn, (l2.activation_fn,)
    # call outputs
    print('check output of encoders')
    src_tokens = batch['src']
    aenc = model.forward_encoder(src_tokens)
    benc = fairseq_model.encoder(src_tokens)['encoder_out'][0]
    print(aenc.shape, benc.shape)
    assert_tensor_eq(aenc, benc)


def run_fairseq_model(fmodel, src, src_len, tgt, encoder_out):
    self = fmodel.decoder
    alignment_layer = None
    prev_output_tokens = tgt
    res = {}
    incremental_state = None
    full_context_alignment = False
    alignment_layer = None
    alignment_heads = None

    if alignment_layer is None:
        alignment_layer = self.num_layers - 1

    # embed positions
    positions = (
        self.embed_positions(
            prev_output_tokens, incremental_state=incremental_state
        )
        if self.embed_positions is not None
        else None
    )

    if incremental_state is not None:
        prev_output_tokens = prev_output_tokens[:, -1:]
        if positions is not None:
            positions = positions[:, -1:]

    # embed tokens and positions
    x = self.embed_scale * self.embed_tokens(prev_output_tokens)

    if self.quant_noise is not None:
        raise RuntimeError('')
        x = self.quant_noise(x)

    if self.project_in_dim is not None:
        raise RuntimeError('')
        x = self.project_in_dim(x)

    if positions is not None:
        x += positions

    if self.layernorm_embedding is not None:
        raise RuntimeError('')
        x = self.layernorm_embedding(x)

    x = self.dropout_module(x)

    # B x T x C -> T x B x C
    x = x.transpose(0, 1)
    res['embedx'] = x

    self_attn_padding_mask: Optional[Tensor] = None
    if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
        self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)
    res['self_attn_padding_mask'] = self_attn_padding_mask
    res['encoder_out'] = encoder_out['encoder_out'][0]
    res['encoder_padding_mask'] = encoder_out['encoder_padding_mask'][0]
    # decoder layers
    attn: Optional[Tensor] = None
    inner_states: List[Optional[Tensor]] = [x]
    for idx, layer in enumerate(self.layers):
        if incremental_state is None and not full_context_alignment:
            self_attn_mask = self.buffered_future_mask(x)
        else:
            self_attn_mask = None
            res['self_attn_mask'] = self_attn_mask

        x, layer_attn, _ = layer(
            x,
            encoder_out["encoder_out"][0]
            if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
            else None,
            encoder_out["encoder_padding_mask"][0]
            if (
                encoder_out is not None
                and len(encoder_out["encoder_padding_mask"]) > 0
            )
            else None,
            incremental_state,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
            need_attn=bool((idx == alignment_layer)),
            need_head_weights=bool((idx == alignment_layer)),
        )
        inner_states.append(x)
        if layer_attn is not None and idx == alignment_layer:
            attn = layer_attn.float().to(x)

    if attn is not None:
        if alignment_heads is not None:
            attn = attn[:alignment_heads]

        # average probabilities over heads
        attn = attn.mean(dim=0)

    if self.layer_norm is not None:
        raise RuntimeError('')
        x = self.layer_norm(x)

    # T x B x C -> B x T x C
    x = x.transpose(0, 1)

    if self.project_out_dim is not None:
        raise RuntimeError('')
        x = self.project_out_dim(x)
    res['inner_states'] = inner_states

    return x, res


def compare_decoder(model, fmodel):
    print('check decoder')
    assert model.decoder.norm is None
    assert fmodel.decoder.layer_norm is None
    for i, (l1, l2) in enumerate(zip_longest(model.decoder.layers, fmodel.decoder.layers)):
        print('check', i, 'th layer of decoder')
        assert l2.normalize_before is False
        compare_multiheadatt(l1.self_attn, l2.self_attn)
        assert_dropout_init_eq(l1.dropout1, l2.dropout_module)
        assert_layernom_init_eq(l1.norm1, l2.self_attn_layer_norm)
        compare_multiheadatt(l1.multihead_attn, l2.encoder_attn)
        assert_dropout_init_eq(l1.dropout2, l2.dropout_module)
        assert_layernom_init_eq(l1.norm2, l2.encoder_attn_layer_norm)
        assert_tensor_init_eq(l1.linear1.weight, l2.fc1.weight)
        assert_tensor_init_eq(l1.linear1.bias, l2.fc1.bias)
        assert_dropout_init_eq(l1.dropout, l2.activation_dropout_module)
        assert_tensor_init_eq(l1.linear2.weight, l2.fc2.weight)
        assert_tensor_init_eq(l1.linear2.bias, l2.fc2.bias)
        assert_dropout_init_eq(l1.dropout3, l2.dropout_module)
        assert_layernom_init_eq(l1.norm3, l2.final_layer_norm)
        assert l1.activation == l2.activation_fn, (l1.activation, l2.activation_fn)
    print('check output layer')
    assert model.project_out is None
    assert fmodel.decoder.project_out_dim is None
    assert_tensor_init_eq(model.output_projection.weight, fmodel.decoder.output_projection.weight)
    assert model.output_projection.bias is None
    assert fmodel.decoder.output_projection.bias is None
    print('check output of decoder')
    print(batch)
    src, tgt = batch['src'], batch['tgt']
    ## 逐步运行transformer和fairse transformer进行检查
    from simdltk.model.transformer import generate_square_subsequent_mask
    amem = model.forward_encoder(src)
    ax, aemb = model.forward_embedding(tgt, model.decoder_embed_tokens,
            model.decoder_embed_positions, model.decoder_embed_dropout, 
            model.decoder_embed_scale, model.decoder_layernorm_embedding
    )
    ax = ax.transpose(0, 1)  # T x B x C
    atgt_mask = generate_square_subsequent_mask(ax.size(0), ax.device)  # T x T
    atgt_key_padding_mask = tgt.eq(model.tgt_vocab.pad())
    amemory_mask = None
    amemory_key_padding_mask = src.eq(model.src_vocab.pad())
    aout, ainner_states = model.decoder(ax, amem, memory_mask=amemory_mask, memory_key_padding_mask=amemory_key_padding_mask, return_inner_states=True,
        tgt_mask=atgt_mask, tgt_key_padding_mask=atgt_key_padding_mask)
    afout = aout
    if model.project_out is not None:
        aout = model.project_out(aout)
    aout = model.output_projection(aout)
    
    bencoder_out = fmodel.encoder(src, batch['src_len'], True)
    assert_tensor_eq(amem, bencoder_out['encoder_out'][0])
    bx, bres = run_fairseq_model(fmodel, src, batch['src_len'], tgt, bencoder_out)
    assert_tensor_eq(ax, bres['embedx'])
    assert_tensor_eq(amemory_key_padding_mask, bres['encoder_padding_mask'])
    print('len states', len(ainner_states), len(bres['inner_states']))
    for i, (ap, bp) in enumerate(zip_longest(ainner_states, bres['inner_states'][1:])):
        print(i, 'th out')
        assert_tensor_eq(ap, bp)
    print(afout.shape, bx.shape)
    assert_tensor_eq(afout.transpose(0, 1), bx)
    # 
    aout = model.forward(src, tgt)['logits']
    bout, _ = fmodel.forward(src, batch['src_len'], tgt)
    print(aout.shape, bout.shape)
    assert_tensor_eq(aout, bout, thr=0.000001)


def compare_model(model, fairseq_model):
    compare_embedding(model, fairseq_model)
    compare_encoder(model, fairseq_model)
    # compare_decoder_layer(model, fairseq_model)
    compare_decoder(model, fairseq_model)


from simdltk.data.dataloader import DataLoader
dl = DataLoader(dsende, batch_size=2, shuffle=False, max_samples_in_memory=1000)
it = iter(dl)
next(it)
batch = next(it) # 选取一个长度不一样的, 带有pad
assert_tensor_init_eq(fmodel.encoder.layers[0].self_attn.out_proj.weight, fmodel.encoder.layers[1].self_attn.out_proj.weight)
model.train()
fmodel.train()
compare_model(model, fmodel)
model.eval()
fmodel.eval()
compare_model(model, fmodel)

# self.encoder_embed_tokens = build_embedding(src_vocab, args.encoder_embed_dim, 
#     path=args.encoder_embed_path
# )
# self.encoder_embed_positions = (
#     PositionalEmbedding(
#         args.max_source_positions,
#         args.encoder_embed_dim,
#         src_vocab.pad(),
#         learned=args.encoder_learned_pos,
#     )
#     if not args.no_token_positional_embeddings
#     else None
# )
# self.encoder_embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(args.encoder_embed_dim)
# self.encoder_layernorm_embedding = nn.LayerNorm(args.encoder_embed_dim) \
#     if args.layernorm_embedding \
#     else None
# self.encoder_embed_dropout = nn.Dropout(args.embed_dropout)

print('OK!')

