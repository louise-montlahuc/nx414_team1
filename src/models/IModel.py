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
        self.ACTs = dict()
        self.PCA = dict()
        self.pca_fitted = []

    def forward(self, images):
        return self.model(images)
    
    def get_layers(self, layer_name=None):
        """
        Returns the layers on which to do the linear probing.
        """
        layers = []
        if not layer_name:
            layers_name = [name for name, _ in self.model.named_children()]
            for name in layers_name[-4:-1]:
                module = self.model.get_submodule(name)
                layers.append((name, module))
        else:
            module = self.model.get_submodule(layer_name)
            layers.append((layer_name, module))
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
        if activations.shape[1] > 1000:
            if layer_name in self.pca_fitted:
                pca_features = self.PCA[layer_name].transform(activations)
                print('Principal components shape:', pca_features.shape)
                self.PCs[layer_name] = pca_features
            else:
                pca = PCA(n_components=1000)
                print(pca)
                self.PCA[layer_name] = pca
                pca_features = pca.fit_transform(activations)
                self.pca_fitted.append(layer_name)
                print('Principal components shape:', pca_features.shape)
                self.PCs[layer_name] = pca_features
        else:
            self.PCs[layer_name] = activations
            
    

    def _get_activations_hook(self, module, input, output, layer_name):
        activations = output.detach().cpu().numpy().reshape(output.shape[0], -1)
        self.ACTs[layer_name] = activations
    
    def register_hook(self, hook_name, finetune, layer_name):
        """
        Registers a hook to the model.
        The hook can be 'all' for all activations or 'pca' for 1000 principal components.
        """
        handles = []
        layers = self.get_layers(layer_name) if finetune else self.get_layers()
            
        for name, layer in layers:
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
        self.insert_after = insert_after
        self.num_classes = num_classes
        
        # Register hook on specified layer
        self._register_hook()

        # Run dummy forward to get shape of features
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            _ = self.model(dummy_input)
            features = self.captured_features

        # Process features according to type (ViT vs CNN)
        if features.dim() == 4:  # CNN style: [B, C, H, W]
            features = nn.AdaptiveAvgPool2d((16, 16))(features)
            features = features.reshape(features.size(0), -1)
        elif features.dim() == 3:  # ViT style: [B, N, D]
            features = features[:, 0]  # Use only [CLS] token
        else:
            raise ValueError("Unknown feature shape from intercepted layer")

        in_features = features.shape[1]

        # Define classifier head
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Flatten(),
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        ) if features.dim() == 4 else  nn.Sequential(
                nn.Linear(in_features, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, num_classes)
            )
        

    def _register_hook(self):
        def hook_fn(module, input, output):
            self.captured_features = output

        # Traverse and hook the desired layer
        for name, module in self.model.named_modules():
            if name == self.insert_after:
                module.register_forward_hook(hook_fn)
                self.hooked_layer = module
                return
        raise ValueError(f"Layer '{self.insert_after}' not found in model.")

    def forward(self, x):
        _ = self.model(x)
        features = self.captured_features

        if features.dim() == 4:  # CNN
            features = nn.AdaptiveAvgPool2d((16, 16))(features)
            features = features.reshape(features.size(0), -1)
        elif features.dim() == 3:  # ViT
            features = features[:, 0]

        return self.fc(features)

    def get_layers(self, layer_name=None):
        return [(self.insert_after, self.hooked_layer)]

