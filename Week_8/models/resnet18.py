from torch import nn
from transformers import AutoImageProcessor, AutoModelForImageClassification

from Week_8.models.Imodel import IModel
from .build import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class ResNet18(IModel):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.model = AutoModelForImageClassification.from_pretrained(
            "microsoft/resnet-18",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")

    def forward(self, images):
        inputs = self.image_processor(images=images, return_tensors="pt")
        logits = self.model(**inputs).logits
        return logits.argmax(dim=-1).item()
    
    def get_layers(self):
        """
        Returns the layers on which to do the linear probing.
        """
        return super().get_layers()