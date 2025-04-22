from abc import ABC, abstractmethod

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

    def forward(self, images):
        return self.model(images)
    
    def replace_head(self, num_classes):
        n_feats = self.model.fc.in_features
        self.model.fc = nn.Linear(n_feats, num_classes)

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
        pca_features = pca.fit_transform(activations)
        print('Principal components shape:', pca_features.shape)
        self.PCs[layer_name] = pca_features

    def _get_activations_hook(self, module, input, output, layer_name):
        activations = output.detach().cpu().numpy().reshape(output.shape[0], -1)
        self.ACTs[layer_name] = activations
    
    def register_hook(self, hook_name):
        """
        Registers a hook to the model.
        The hook can be 'all' for all activations or 'pca' for 1000 principal components.
        """
        handles = []
        for name, layer in self.get_layers():
            if hook_name == 'all':
                handle = layer.register_forward_hook(lambda m, i, o, n=name: self._get_activations_hook(m, i, o, n))
            elif hook_name == 'pca':
                handle = layer.register_forward_hook(lambda m, i, o, n=name: self._get_PCs_hook(m, i, o, n))
            handles.append(handle)
        return handles

    @abstractmethod
    def get_layers(self):
        """
        Returns the layers of the model.
        By default, it returns the last layer of the model.
        """
        pass