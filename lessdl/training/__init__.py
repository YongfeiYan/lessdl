import os
import importlib
import functools

from lessdl.training.utils import load_exp_args, move_to_device, load_args

TRAINERS = {}
CALLBACKS = {}


def register_trainer(name):
    """
    @register_trainer('trainer_name')
    class xxx
        ...
    """
    def wrapper(cls):
        assert name not in TRAINERS, f'{name} is already registered for {cls}'
        TRAINERS[name] = cls
        return cls
    return wrapper


def get_trainer_cls(name):
    assert name in TRAINERS, f'Trainer {name} is not registered, availables are ({list(TRAINERS.keys())})'
    return TRAINERS[name]


def register_callback(name):
    """
    @register_callback('callback_name')
    class xxx
    """
    def wrapper(cls):
        assert name not in CALLBACKS, f'Callback {name} is already registered for {cls}'
        CALLBACKS[name] = cls
        return cls
    return wrapper


def get_callback_cls(name):
    assert name in CALLBACKS, 'Callback {} is not found, availables are {}'.format(name, list(CALLBACKS.keys()))
    return CALLBACKS[name]


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
        module = importlib.import_module("lessdl.training." + model_name)


