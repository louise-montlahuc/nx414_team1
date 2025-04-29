from torch import nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

from models.IModel import IModel
from models.build import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class ConvNeXt(IModel):
    def __init__(self):
        super(ConvNeXt, self).__init__()
        self.model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
    
    def get_layers(self):
        """
        Returns the layers on which to do the linear probing.
        """
        layer7 = list(self.model.children())[-1]
        print(layer7)
        classif = self.model.get_submodule("classifier")
        return [('layer7', layer7), ('classifier', classif)]
    
    def replace_head(self, num_classes):
        # TODO
        self.fc = nn.Linear(self.model.classifier.in_features, num_classes)
        self.model.classifier = self.fc