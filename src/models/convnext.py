from torchvision.models import convnext_base, ConvNeXt_Base_Weights

from models.IModel import IModel
from models.build import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class ConvNeXt(IModel):
    def __init__(self):
        super(ConvNeXt, self).__init__()
        self.model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
    
    def get_layers(self, driven):
        """
        Returns the layers on which to do the linear probing.
        """    
        layer7 = self.model.get_submodule("features.7")
        classif = self.model.get_submodule("classifier")
        return [('layer7', layer7), ('classifier', classif)]