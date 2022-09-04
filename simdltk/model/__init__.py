import os
import importlib
import functools


MODELS = {}
ARCH_MODEL = {}
ARCH_FUNC = {}


def register_model(name):
    """
    @register_model('model_name')
    class xxx

    """
    def wrapper(cls):
        assert name not in MODELS, f'{name} is already registered for {cls}'
        MODELS[name] = cls
        return cls
    return wrapper


def get_arch_model(name):
    return ARCH_MODEL.get(name, None)


def get_arch_arch(name):
    assert name in ARCH_FUNC, f'Arch {name} is not registered, available choices are {list(ARCH_FUNC.keys())}'
    return ARCH_FUNC[name]


def get_model_cls(name=None, arch=None):
    assert name is not None or arch is not None
    if arch is not None:
        model_cls = get_arch_model(arch)
        assert model_cls is not None, f'Architecture {arch} is not registered.'
        return model_cls
    assert name in MODELS, f'Model {name} is not registered in models ({list(MODELS.keys())})'
    return MODELS[name]


def register_model_architecture(model_name, arch_name):
    """
    @register_model_architecture('model', 'arch')
    def xxxx_arch(args):
        modify arguments in args
    """
    def wrapper(fn):
        assert model_name in MODELS, f'{model_name} is not registered.'
        assert arch_name not in ARCH_MODEL, f'{arch_name} is already registered.'
        ARCH_MODEL[arch_name] = MODELS[model_name]
        ARCH_FUNC[arch_name] = fn
        return fn

    return wrapper


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
        module = importlib.import_module("simdltk.model." + model_name)



