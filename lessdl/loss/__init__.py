import os
import importlib
from torch.nn import Module

_LOSS = {}


def register_loss(name):
    assert name not in _LOSS, f"{name} is already registered."
    def wrapper(cls):
        _LOSS[name] = cls
        return cls
    return wrapper


def get_loss_cls(name):
    assert name in _LOSS, f'Loss {name} is not registered.'
    return _LOSS[name]


class Loss(Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def add_args(parser, arglist=None):
        pass
    
    @classmethod
    def build(cls, args, model, dataset):
        return cls()

    def forward(self, *args, **kwargs):
        """
        返回一个dict, 其中有loss作为key
        """
        raise NotImplementedError('')

    def reduce(self, losses):
        """
        将多次计算的loss累积, 得到一个新的loss. 
        需要多个batch进行计算loss的时候会用到这个函数, 否则可以不必实现.
        """
        raise NotImplementedError('')


@register_loss('NoopLoss')
class NoopLoss(Loss):
    def forward(self, *args, **kwargs):
        return {}


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("lessdl.loss." + model_name)

