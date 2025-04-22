from .build import MODEL_REGISTRY, make_model
from .resnet18 import ResNet18
from .resnet50 import ResNet50
from .resnext import ResNeXt
from .vit import ViT
from .convnext import ConvNeXt
from .dinov2 import DinoV2
from . import linear_reg  
from . import ridge_reg  
from . import mlp_reg  