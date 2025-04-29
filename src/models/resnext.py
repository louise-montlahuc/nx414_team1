from torch import nn
from torchvision.models import resnext101_32x8d, ResNeXt101_32X8D_Weights

from models.IModel import IModel
from models.build import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class ResNeXt(IModel):
    def __init__(self):
        super(ResNeXt, self).__init__()
        self.model = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
    
    def get_layers(self):
        """
        Returns the layers on which to do the linear probing.
        """
        module = self.model.get_submodule("layer4")
        return [('layer4', module)]
    
    def replace_head(self, num_classes):
        # TODO
        self.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.fc = self.fc