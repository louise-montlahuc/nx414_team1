from torchvision.models import resnet18, ResNet18_Weights

from models.IModel import IModel
from models.build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class ResNet18(IModel):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
@MODEL_REGISTRY.register()
class ResNet18_randomW(IModel):
    def __init__(self):
        super(ResNet18_randomW, self).__init__()
        self.model = resnet18(weights=None)
    
    
    
