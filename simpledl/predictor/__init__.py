import os
import importlib


PREDICTOR = {}


def register_predictor(name):
    """
    @register_predictor('predictor_name')
    class xxx

    """
    def wrapper(cls):
        assert name not in PREDICTOR, f'Predictor {name} is already registered for {cls}'
        PREDICTOR[name] = cls
        return cls
    return wrapper


def get_predictor_cls(name):
    assert name in PREDICTOR, f'Predictor {name} is not registered, available choices are {list(PREDICTOR.keys())}'
    return PREDICTOR[name]




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
        module = importlib.import_module("simpledl.predictor." + model_name)






























