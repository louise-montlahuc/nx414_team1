import torch
import inspect
from fvcore.common.registry import Registry

from models.IModel import IModel

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for models.

The registered object will be called with `obj()`.
The call should return a `nn.Module` object.
"""


def accepts_seed(cls):
    init = cls.__init__
    sig = inspect.signature(init)
    return 'seed' in sig.parameters

def make_model(name, seed):
    """
    Builds the video model.
    Args:
        name (string): name of the model to build.
    Returns:
        model (nn.Module): the built model.
    """
    model = MODEL_REGISTRY.get(name)
    if accepts_seed(model):
        model = model(seed)
    else:
        model = model()
    
    return model