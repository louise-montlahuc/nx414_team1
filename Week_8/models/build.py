import torch
from fvcore.common.registry import Registry

import Imodel

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for models.

The registered object will be called with `obj()`.
The call should return a `nn.Module` object.
"""

def build_model(name):
    """
    Builds the video model.
    Args:
        name (string): name of the model to build.
    Returns:
        model (nn.Module): the built model.
    """
    model = MODEL_REGISTRY.get(name)()

    if isinstance(model, Imodel) and torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
    
    return model