from abc import ABC, abstractmethod

from torch import nn
from sklearn.decomposition import PCA
class IModel(ABC, nn.Module):
    """
    Abstract base class for a model.
    This class defines the interface that all model classes must implement.
    """
    def __init__(self):
        super().__init__()
        self.PCs = dict()
        self.ACTs = dict()

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
        print('Layer:', layer_name)
        activations = output.detach().cpu().numpy().reshape(output.shape[0], -1)
        print('Activations shape:', activations.shape)
        self.ACTs[layer_name] = activations
    
    def register_hook(self, hook_name):
        """
        Registers a hook to the model.
        The hook can be 'all' for all activations or 'pca' for 1000 principal components.
        """
        print(self.get_layers())
        for name, layer in self.get_layers():
            if hook_name == 'all':
                layer.register_forward_hook(lambda m, i, o, n=name: self._get_activations_hook(m, i, o, n))
            elif hook_name == 'pca':
                layer.register_forward_hook(lambda m, i, o, n=name: self._get_PCs_hook(m, i, o, n))

    @abstractmethod
    def get_layers(self):
        """
        Returns the layers of the model.
        By default, it returns the last layer of the model.
        """
        pass