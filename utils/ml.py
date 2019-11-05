import importlib


def load_model_def(model_name):
    model = getattr(importlib.import_module("ml.models."+model_name), "model")
    return model
