from abc import ABC

import torch
from torch import nn
from sklearn.decomposition import PCA

class IModel(ABC, nn.Module):
    """
    Abstract base class for a model.
    This class defines the interface that all model classes must implement.
    All models inheriting from this class should have a self.model attribute!
    """
    def __init__(self):
        super().__init__()
        self.PCs = dict()
        self.PCA = None
        self.ACTs = dict()

    def forward(self, images):
        return self.model(images)
    
    def get_layers(self, driven):
        """
        Returns the layers on which to do the linear probing.
        """
        layers = []
        layers_name = [name for name, _ in self.model.named_children()]
        for name in layers_name[-4:-1]:
            module = self.model.get_submodule(name)
            layers.append((name, module))
        return layers
        
    def get_activations(self, hook_name):
        """
        Returns the activations of the model.
        The hook_name can be 'all' for all activations or 'pca' for 1000 principal components.
        """
        if hook_name == 'all':
            return self.ACTs
        elif hook_name == 'pca':
            return self.PCs
        else:
            raise ValueError("Invalid hook name. Use 'all' or 'pca'.")
        
    def reset_activations(self):
        """
        Resets the activations of the model.
        """
        self.PCs = dict()
        self.ACTs = dict()

    def _get_PCs_hook(self, module, input, output, layer_name):
        print('Layer:', layer_name)
        activations = output.detach().cpu().numpy().reshape(output.shape[0], -1)
        print('Activations shape:', activations.shape)
        pca = PCA(n_components=1000)
        print(pca.type())
        self.PCA = pca
        pca_features = pca.fit_transform(activations)
        print('Principal components shape:', pca_features.shape)
        self.PCs[layer_name] = pca_features

    def _get_activations_hook(self, module, input, output, layer_name):
        activations = output.detach().cpu().numpy().reshape(output.shape[0], -1)
        self.ACTs[layer_name] = activations
    
    def register_hook(self, hook_name, driven):
        """
        Registers a hook to the model.
        The hook can be 'all' for all activations or 'pca' for 1000 principal components.
        """
        handles = []
        for name, layer in self.get_layers(driven):
            if hook_name == 'all':
                handle = layer.register_forward_hook(lambda m, i, o, n=name: self._get_activations_hook(m, i, o, n))
            elif hook_name == 'pca':
                handle = layer.register_forward_hook(lambda m, i, o, n=name: self._get_PCs_hook(m, i, o, n))
            handles.append(handle)
        return handles
    
    def change_head(self, layer, num_classes):
        """
        Sets a final head (classification or regression) after the indicated layer.
        """
        return ModifiedModel(self.model, layer, num_classes)


class ModifiedModel(IModel):
    def __init__(self, base_model, insert_after, num_classes):
        super().__init__()
        self.model = base_model
        self.base_model = base_model
        self.insert_after = insert_after
        self.num_classes = num_classes
        self.layer_name = insert_after

        # Extract layers up to the insertion point
        self.features = nn.Sequential()
        for name, module in base_model.named_children():
            if isinstance(module, nn.ModuleList):
                for subname, submodule in module.named_children():
                    self.features.add_module(subname, submodule)
                    if f'{name}.{subname}' == insert_after:
                        self.layer = (name, module)
                        break
                else:
                    continue
                break
            else:
                self.features.add_module(name, module)
                if name == insert_after:
                    self.layer = (name, module)
                    break

        # Determine input dim for new head
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self._forward_features(dummy_input)
            pooled = nn.AdaptiveAvgPool2d((16, 16))(features)
            flat = pooled.view(pooled.size(0), -1)
            head_in_features = flat.shape[1]  

        # Define new head (classification or regression)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Flatten(),
            nn.Linear(head_in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def _forward_features(self, x):
        for name, module in self.features.named_children():
            if isinstance(module, nn.ModuleList):
                for submodule in module:
                    x = submodule(x)
            else:
                x = module(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.fc(x)
        return x
    
    def get_layers(self, driven):
        if driven == 'data':
            return [(self.layer_name, self.layer)]

        return super().get_layers(driven)