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
        layers = []
        layers_name = [name for name, _ in self.model.named_children()]
        for name in layers_name[-4:]:
            module = self.model.get_submodule(name)
            layers.append((name, module))
        return layers 