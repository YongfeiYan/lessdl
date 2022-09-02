"""
来自fairseq的criterion
"""
import torch
import math
from torch.nn import functional as F

from simdltk.loss import register_loss, Loss
from simdltk.utils import bool_flag


# def get_perplexity(loss, base=2):
#     if loss is None:
#         return 0.0
#     try:
#         return base ** loss
#     except OverflowError:
#         return float("inf")


def get_log_prob_target(batch, out):
    if 'logits' in out:
        log_probs = F.log_softmax(out['logits'], dim=-1)
    else:
        assert 'log_probs' in out, f'logits or log_probs should be in output of moels, but found {list(out.keys())}'
        log_probs = out['log_probs']
    assert 'target' in batch, f'target should be in batch of data, found {list(batch.keys())}.'
    target = batch['target']
    return log_probs, target


@register_loss('binary_cross_entropy')
class BinaryCrossEntropy(Loss):
    def __init__(self, reduction):
        super().__init__()
        assert reduction in ['sum', 'mean']
        self.reduction = reduction
    
    @staticmethod
    def add_args(parser, arglist=None):
        parser.add_argument('--reduction', type=str, default='mean')

    @classmethod
    def build(self, args, model, dataset):
        return BinaryCrossEntropy(args.reduction)
    
    def forward(self, batch, out):
        probs = out['probs']
        target = batch['target'].float()
        loss = F.binary_cross_entropy(probs, target, reduction=self.reduction)
        batch_size = len(target)
        return {
            'loss': loss,
            'sample_size': batch_size,
            'sample_loss': loss,
        }


@register_loss('cross_entropy')
class CrossEntropy(Loss):
    def __init__(self, padding_idx, sentence_avg):
        super().__init__()
        self.sentence_avg = sentence_avg
        self.padding_idx = padding_idx

    @classmethod
    def build(cls, args, model, dataset):
        if args.cross_entropy_padding_idx is None:
            padding_idx = dataset.padding_idx
        else:
            padding_idx = args.cross_entropy_padding_idx
        sentence_avg = args.sentence_avg
        return cls(padding_idx, sentence_avg)

    @staticmethod
    def add_args(parser, arglist=None):
        parser.add_argument('--sentence-avg', action='store_true')
        parser.add_argument('--cross-entropy-padding-idx', type=int, default=None)

    def forward(self, batch, out):
        """
        batch 和 model output
        TODO: 将ppl加进去.
        """
        log_probs, target = get_log_prob_target(batch, out)
        loss = F.nll_loss(
            log_probs.view(-1, log_probs.size(-1)), 
            target.view(-1), 
            ignore_index=self.padding_idx,
            reduction='sum', 
        )
        ntokens = target.ne(self.padding_idx).long().sum().item()
        sample_size = target.size(0) if self.sentence_avg else ntokens
        return {
            'loss': loss / sample_size,
            'sample_loss': loss,
            'sample_size': sample_size,
            'ntokens': ntokens
        }

    def reduce(self, losses):
        sample_loss = torch.sum([a['sample_loss'] for a in losses])
        sample_size = torch.sum([a['sample_size'] for a in losses])
        ntokens = torch.sum([a['ntokens'] for a in losses])
        return {
            'loss': sample_loss / sample_size,
            'sample_loss': sample_loss,
            'sample_size': sample_size,
            'ntokens': ntokens
        }

    # @staticmethod
    # def reduce_metrics(logging_outputs) -> None:
    #     """Aggregate logging outputs from data parallel training."""
    #     loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
    #     ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
    #     sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

    #     # we divide by log(2) to convert the loss from base e to base 2
    #     metrics.log_scalar(
    #         "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
    #     )
    #     if sample_size != ntokens:
    #         metrics.log_scalar(
    #             "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
    #         )
    #         metrics.log_derived(
    #             "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
    #         )
    #     else:
    #         metrics.log_derived(
    #             "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
    #         )


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    """
    返回:
        smoothed loss, 和nll loss. 如果reduce的话, 返回各个元素loss的和, loss.sum()
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_loss("label_smoothed_cross_entropy")
class LabelSmoothedCrossEntropy(Loss):
    def __init__(
        self,
        padding_idx,
        sentence_avg,
        label_smoothing,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing

    @classmethod
    def build(cls, args, model, dataset):
        padding_idx = dataset.padding_idx
        sentence_avg = args.sentence_avg
        label_smoothing = args.label_smoothing
        assert args.label_smoothing is not None
        return cls(padding_idx, sentence_avg, label_smoothing)

    @staticmethod
    def add_args(parser, arglist=None):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--sentence-avg', action='store_true')
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # parser.add_argument('--report-accuracy', action='store_true',
        #                     help='report accuracy metric')
        # parser.add_argument('--ignore-prefix-size', default=0, type=int,
        #                     help='Ignore first N tokens')
        # fmt: on

    def forward(self, batch, out):
        """Compute the loss for the given sample.
        model, sample, reduce=True
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        log_probs, target = get_log_prob_target(batch, out)
        log_probs = log_probs.view(-1, log_probs.size(-1))
        target = target.view(-1)
        ntokens = target.ne(self.padding_idx).long().sum().item()
        sample_size = target.size(0) if self.sentence_avg else ntokens
        loss, nll_loss = label_smoothed_nll_loss(log_probs, target, self.eps, ignore_index=self.padding_idx, reduce=True)
        return {
            'loss': loss / sample_size,
            'sample_loss': loss,
            'sample_size': sample_size,
            'ntokens': ntokens,
            'sample_nll_loss': nll_loss,
            'nll_loss': nll_loss / sample_size,
        }

    def reduce(self, losses):
        sample_loss = torch.sum([t['sample_loss'] for t in losses])
        sample_size = torch.sum([t['sample_size'] for t in losses])
        ntokens = torch.sum([t['ntokens'] for t in losses])
        sample_nll_loss = torch.sum([t['sample_nll_loss'] for t in losses])
        return {
            'loss': sample_loss / sample_size,
            'sample_loss': sample_loss,
            'sample_size': sample_size,
            'ntokens': ntokens,
            'sample_nll_loss': sample_nll_loss,
            'nll_loss': sample_nll_loss / sample_size,
        }
