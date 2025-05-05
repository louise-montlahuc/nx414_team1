import torch

from models.IModel import IModel
from models.build import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class DinoV2(IModel):
    def __init__(self):
        super(DinoV2, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        
    def get_layers(self, layer_name=None):
        """
        Returns the layers on which to do the linear probing.
        """    
        module_block11 = self.model.get_submodule("blocks.11")
        module_norm = self.model.get_submodule("norm")
        if layer_name == "blocks.11":
            return [('block11', module_block11)]
        elif layer_name == "norm":
            return [('norm', module_norm)]
        else:
            return [('block11', module_block11), ('norm', module_norm)]
