from torchvision.models import resnet50, ResNet50_Weights

from models.IModel import IModel
from models.build import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class ResNet50(IModel):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    
    def get_layers(self):
        """
        Returns the layers on which to do the linear probing.
        """
        module_layer4 = self.model.get_submodule("layer4")
        module_avgpool = self.model.get_submodule("avgpool")
        return [('layer4', module_layer4), ('avgpool', module_avgpool)]