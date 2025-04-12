from abc import ABC, abstractmethod

class IModel(ABC, nn.Module):
    """
    Abstract base class for a model.
    This class defines the interface that all model classes must implement.
    """

    @abstractmethod
    def get_layers(self):
        """
        Returns the layers of the model.
        By default, it returns the last layer of the model.
        """
        return self.model.children()[-1]