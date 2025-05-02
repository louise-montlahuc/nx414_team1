from torchvision.models import vit_b_16, ViT_B_16_Weights

from models.IModel import IModel
from models.build import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class ViT(IModel):
    def __init__(self):
        super(ViT, self).__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    
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