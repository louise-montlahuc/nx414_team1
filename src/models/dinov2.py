import torch

from models.IModel import IModel
from models.build import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class DinoV2(IModel):
    def __init__(self):
        super(DinoV2, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    
    def get_layers(self):
        """
        Returns the layers on which to do the linear probing.
        """
        module = self.model.get_submodule("norm")
        return [('norm', module)]