import os
import math
from torch import nn

from lessdl.module.transformer import Transformer as BaseTransformer, TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer, generate_square_subsequent_mask, generate_timestep_mask
from lessdl.model.base import EncDecModel
from lessdl.module.embedding import build_embedding, PositionalEmbedding
from lessdl.model import register_model, register_model_architecture
from lessdl.module import update_prev_state, extract_sub_state
from lessdl.utils import bool_flag

DEFAULT_MAX_POSITIONS = 1024


@register_model('transformer')
class Transformer(EncDecModel):
    def __init__(self, args, src_vocab, tgt_vocab):
        super().__init__(args, src_vocab, tgt_vocab)
        # encoder embeddings
        self.encoder_embed_tokens = build_embedding(src_vocab, args.encoder_embed_dim, 
            path=args.encoder_embed_path
        )
        self.encoder_embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                args.encoder_embed_dim,
                src_vocab.pad(),
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )
        self.encoder_embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(args.encoder_embed_dim)
        self.encoder_layernorm_embedding = nn.LayerNorm(args.encoder_embed_dim) \
            if args.layernorm_embedding \
            else None
        self.encoder_embed_dropout = nn.Dropout(args.embed_dropout)

        # encoder
        encoder_layer = TransformerEncoderLayer(args.encoder_embed_dim, args.encoder_attention_heads, 
            args.encoder_ffn_embed_dim, args.dropout, args.attention_dropout, args.activation_fn
        )
        encoder_layernorm = None
        self.encoder = TransformerEncoder(encoder_layer, args.encoder_layers, encoder_layernorm)

        # decoder embeddings
        if args.share_all_embeddings:
            assert src_vocab == tgt_vocab, 'share all embeddings : src_vocab == tgt_vocab'
            assert args.encoder_embed_dim == args.decoder_embed_dim
            assert args.encoder_embed_path == args.decoder_embed_path
            assert args.encoder_learned_pos == args.decoder_learned_pos
            self.decoder_embed_tokens = self.encoder_embed_tokens
            self.decoder_embed_positions = self.encoder_embed_positions
            self.decoder_embed_scale = self.encoder_embed_scale
            self.decoder_layernorm_embedding = self.encoder_layernorm_embedding
            self.decoder_embed_dropout = self.encoder_embed_dropout
        else:
            self.decoder_embed_tokens = build_embedding(tgt_vocab, args.decoder_embed_dim, args.decoder_embed_path)
            self.decoder_embed_positions = (
                PositionalEmbedding(
                    args.max_target_positions,
                    args.decoder_embed_dim,
                    tgt_vocab.pad(),
                    learned=args.decoder_learned_pos
                )
                if not args.no_token_positional_embeddings
                else None
            )
            self.decoder_embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(args.decoder_embed_dim)
            self.decoder_layernorm_embedding = nn.LayerNorm(args.decoder_embed_dim) \
                if args.layernorm_embedding \
                else None
            self.decoder_embed_dropout = nn.Dropout(args.embed_dropout)

        # decoder
        decoder_layer = TransformerDecoderLayer(args.decoder_embed_dim, args.decoder_attention_heads, 
            args.decoder_ffn_embed_dim, args.dropout, args.attention_dropout, args.activation_fn)
        decoder_layernorm = None
        self.decoder = TransformerDecoder(decoder_layer, args.decoder_layers, decoder_layernorm)
        # output
        if args.decoder_embed_dim != args.decoder_output_dim:
            self.project_out = nn.Linear(args.decoder_embed_dim, args.decoder_output_dim, bias=False)
            nn.init.xavier_uniform_(self.project_out.weight)
        else:
            self.project_out = None
        if args.share_input_output_embed:
            self.output_projection = nn.Linear(
                args.decoder_embed_dim,
                len(tgt_vocab),
                bias=False
            )
            self.output_projection.weight = self.decoder_embed_tokens.weight
        else:
            self.output_projection = nn.Linear(args.decoder_output_dim, len(tgt_vocab), bias=False)
            nn.init.normal_(self.output_projection.weight, mean=0, std=args.decoder_output_dim ** -0.5)

    def forward_embedding(self, tokens, embed_tokens, embed_positions, dropout,
        embed_scale=1.0, layernorm_embedding=None, timestep=None):
        """
        x: 包括了dropout和可能的layernorm
        embed: embeding + token embedding
        """
        embed = embed_tokens(tokens)
        x = embed = embed * embed_scale
        if embed_positions is not None:
            x = embed + embed_positions(tokens, timestep=timestep)
        if layernorm_embedding is not None:
            x = layernorm_embedding(x)
        x = dropout(x)
        return x, embed

    def forward_encoder(self, src):
        """
        src: B x S
        Return
            memory: S x B x C
        """
        # B x S x C
        x, embed = self.forward_embedding(src, self.encoder_embed_tokens, 
            self.encoder_embed_positions, self.encoder_embed_dropout, 
            self.encoder_embed_scale, self.encoder_layernorm_embedding)
        x = x.transpose(0, 1)  # S x B x C
        src_key_padding_mask = src.eq(self.src_vocab.pad())  # B x S
        src_mask = None
        return self.encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

    def forward_decoder(self, memory, src, tgt, prev_state=None):
        """
        out: T x B x _
        
        如果prev_state != None, tgt只有一个时刻的, shape为 B x 1
        """
        # bsz = src.size(0)

        if prev_state is None:
            new_state = None
            timestep = None
            sub_state = None
        else:
            new_state = {}
            timestep = prev_state.get('timestep', tgt.new_zeros((tgt.size(0), 1)))
            new_state['timestep'] = timestep + 1
            sub_state = extract_sub_state(prev_state, 'decoder.')

        # B x T x C
        x, embed = self.forward_embedding(tgt, self.decoder_embed_tokens, 
            self.decoder_embed_positions, self.decoder_embed_dropout, 
            self.decoder_embed_scale, self.decoder_layernorm_embedding, timestep=timestep)
        x = x.transpose(0, 1)  # T x B x C
        if prev_state is None:
            tgt_mask = generate_square_subsequent_mask(x.size(0), x.device)  # T x T
        else:
            tgt_mask = tgt.new_ones((1, timestep[0, 0].item() + 1))
        tgt_key_padding_mask = tgt.eq(self.tgt_vocab.pad())
        memory_mask = None
        memory_key_padding_mask = src.eq(self.src_vocab.pad())
        out = self.decoder(x, memory, 
            memory_mask=memory_mask, memory_key_padding_mask=memory_key_padding_mask,
            tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, prev_state=sub_state,
        )
        if prev_state is not None:
            out, sub_state = out
            update_prev_state(new_state, 'decoder.', sub_state)
        if self.project_out is not None:
            out = self.project_out(out)
        out = self.output_projection(out)
        return out, new_state

    def forward(self, src, tgt, prev_state=None):
        """
        src: B x S
        tgt: B x T
        prev_state: 当循环输入的时候, 用dict记录状态, 并且每个key对应的value第一维度都是batch. 先不考虑, 到beam search的时候再考虑.
        Return:
            out: B x T x |V| logits
        """
        if prev_state is None or 'memory' not in prev_state:
            memory = self.forward_encoder(src)  # S x B x _
        else:
            memory = prev_state['memory'].transpose(0, 1)
        out, prev_state = self.forward_decoder(memory, src, tgt, prev_state)  # T x B x _
        
        if prev_state is not None:
            prev_state['memory'] = memory.transpose(0, 1)

        ret = {
            'memory': memory.transpose(0, 1),  # B x S x _
            'logits': out.transpose(0, 1),  # B x T x |V|
        }
        if prev_state is None:
            return ret
        else:
            return ret, prev_state

    @staticmethod
    def add_args(parser, arglist=None):
        # encoder embedding
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
            help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', type=str, default=None,
            help='embedding weights path')
        parser.add_argument('--no-scale-embedding', action='store_true')
        parser.add_argument('--layernorm-embedding', action='store_true')
        parser.add_argument('--embed-dropout', type=float, metavar='D',
            help='dropout of embeddings.')
        parser.add_argument('--no-token-positional-embeddings', action='store_true')
        parser.add_argument('--max-source-positions', type=int, metavar='N', default=DEFAULT_MAX_POSITIONS,
            help='for positional embeddings')
        parser.add_argument('--encoder-learned-pos', action='store_true',
            help='use learned positional embeddings in the encoder')
        # encoder
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
            help='num encoder attention heads')
        # parser.add_argument('--encoder-normalize-before', action='store_true',
        #     help='apply layernorm before each encoder block')
        # parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
        #     help='LayerDrop probability for encoder')
        parser.add_argument('--activation-fn', choices=['relu', 'gelu'],
            help='activation function')
        parser.add_argument('--dropout', type=float, metavar='D',
            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D', help='在selfattn中用到的dropout')
        # decoder embedding
        parser.add_argument('--share-all-embeddings', action='store_true')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='D')
        parser.add_argument('--decoder-embed-path', type=str, default=None,
            help='embedding weights path')
        parser.add_argument('--share-input-output-embed', type=bool_flag, default=None, 
            help='share decoder input and output embed')
        parser.add_argument('--max-target-positions', type=int, metavar='N', default=DEFAULT_MAX_POSITIONS)
        parser.add_argument('--decoder-learned-pos', action='store_true')
        # decoder
        parser.add_argument('--decoder-layers', type=int, metavar='N')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N')
        parser.add_argument('--decoder-attention-heads', type=int)
        # output
        parser.add_argument('--decoder-output-dim', type=int)


@register_model_architecture("transformer", "transformer")
def base_architecture(args):
    # 下面的参数来自fairseq
    # args.encoder_embed_path = getattr(args, encoder_embed_path", None)
    # args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_embed_dim = args.encoder_embed_dim or 512
    # args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_ffn_embed_dim = args.encoder_ffn_embed_dim or 2048
    # args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_layers = args.encoder_layers or 6
    # args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_attention_heads = args.encoder_attention_heads or 8
    # args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    # args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    # args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = args.decoder_embed_dim or args.encoder_embed_dim
    args.decoder_ffn_embed_dim = args.decoder_ffn_embed_dim or args.encoder_ffn_embed_dim
    args.decoder_layers = args.decoder_layers or 6
    args.decoder_attention_heads = args.decoder_attention_heads or 8
    # args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    # args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    # args.activation_dropout = args.activation_dropout or 0.0
    args.activation_fn = args.activation_fn or "relu"
    args.attention_dropout = args.attention_dropout or 0.0
    args.dropout = args.dropout or 0.1
    # args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    # args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    # args.share_input_output_embed = getattr(
    #     args, "share_input_output_embed", False
    # )
    args.share_input_output_embed = True if args.share_input_output_embed is None else args.share_input_output_embed
    # args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    # args.no_token_positional_embeddings = getattr(
    #     args, "no_token_positional_embeddings", False
    # )
    # args.adaptive_input = getattr(args, "adaptive_input", False)
    # args.no_cross_attention = getattr(args, "no_cross_attention", False)
    # args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = args.decoder_output_dim or args.decoder_embed_dim
    # args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    # args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    # args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    # args.checkpoint_activations = getattr(args, "checkpoint_activations", False)

    # args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    # args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    # args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    # args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    # args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    # args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    # args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

    # 我添加的
    args.embed_dropout = args.embed_dropout or args.dropout
    return args


@register_model_architecture("transformer", "transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = args.encoder_embed_dim or 512
    args.encoder_ffn_embed_dim = args.encoder_ffn_embed_dim or 1024
    args.encoder_attention_heads = args.encoder_attention_heads or 4
    args.encoder_layers = args.encoder_layers or 6
    args.decoder_embed_dim = args.decoder_embed_dim or 512
    args.decoder_ffn_embed_dim = args.decoder_ffn_embed_dim or 1024
    args.decoder_attention_heads = args.decoder_attention_heads or 4
    args.decoder_layers = args.decoder_layers or 6
    base_architecture(args)
    return args
