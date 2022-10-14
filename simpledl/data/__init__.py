import os
import importlib


DATASET = {}


def register_dataset(name):
    assert name not in DATASET, f'{name} is already registered.'

    def wrapper(cls):
        DATASET[name] = cls
        return cls
    
    return wrapper


def get_dataset_cls(name):
    assert name in DATASET, 'Dataset {} is not registered. The following datasets are available {}'.format(name, sorted(list(DATASET.keys())))
    return DATASET[name]


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
        module = importlib.import_module("simpledl.data." + model_name)

