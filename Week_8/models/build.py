import torch
from fvcore.common.registry import Registry

from models.IModel import IModel

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for models.

The registered object will be called with `obj()`.
The call should return a `nn.Module` object.
"""

def make_model(name):
    """
    Builds the video model.
    Args:
        name (string): name of the model to build.
    Returns:
        model (nn.Module): the built model.
    """
    device = 'cpu'
    model = MODEL_REGISTRY.get(name)()

    if isinstance(model, IModel) and torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
    
    return model, device