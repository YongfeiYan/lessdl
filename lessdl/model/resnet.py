
from torch import nn 
from torchvision import models

from lessdl.utils import bool_flag
from lessdl.model import register_model


@register_model('torchvision_models')
class TorchVisionModel(nn.Module):
    def __init__(self, model_name, pretrained):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        model_name = getattr(models, model_name, None)
        assert model_name, 'model_name {} is not found in torchvision.models'.format(model_name)
        self.model = model_name(pretrained=pretrained)
    
    def forward(self, x):
        return {
            'logits': self.model(x)
        }

    @staticmethod
    def add_args(parser, arglist=None):
        parser.add_argument('--model-name', type=str, required=True)
        parser.add_argument('--pretrained', type=bool_flag, default=False)

    @classmethod
    def build(cls, args, dataset):
        return cls(args.model_name, args.pretrained)
