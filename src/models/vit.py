from torchvision.models import vit_b_16, ViT_B_16_Weights

from models.IModel import IModel
from models.build import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class ViT(IModel):
    def __init__(self):
        super(ViT, self).__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
@MODEL_REGISTRY.register()
class ViT_randomW(IModel):
    def __init__(self):
        super(ViT_randomW, self).__init__()
        self.model = vit_b_16(weights=None)
    