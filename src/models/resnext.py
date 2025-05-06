from torchvision.models import resnext101_32x8d, ResNeXt101_32X8D_Weights

from models.IModel import IModel
from models.build import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class ResNeXt(IModel):
    def __init__(self):
        super(ResNeXt, self).__init__()
        self.model = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
        
        
@MODEL_REGISTRY.register()
class ResNeXt_randomW(IModel):
    def __init__(self):
        super(ResNeXt_randomW, self).__init__()
        self.model = resnext101_32x8d(weights=None)

