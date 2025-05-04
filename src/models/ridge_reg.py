import torch
from sklearn.linear_model import Ridge

from models.IModel import IModel
from models.build import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class ridge_reg(IModel):
    def __init__(self, seed):
        super(ridge_reg, self).__init__()
        self.model = Ridge(alpha=1.0, random_state=seed)
        self.ACTs = {} 

    def forward(self, images):
        self.ACTs = {'ridge_regression': images}
        return self.model.predict(images)

    def fit(self, x, y):
        """ Fit the regression model using the provided method """
        self.model.fit(x, y)
        return self.model
    
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            return self(X)
    
    def get_layers(self, driven):
        return [('ridge_regression', None)]
    
    def get_activations(self, layer):
        print("Returning activations:", self.ACTs.keys())
        return self.ACTs

    def register_hook(self, hook_name, driven):
        print(f"Linear model does not support hooks. Hook name: {hook_name}")
        return []