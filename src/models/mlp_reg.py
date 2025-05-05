import torch
from sklearn.neural_network import MLPRegressor
from sklearn.discriminant_analysis import StandardScaler

from models.IModel import IModel
from models.build import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class mlp_reg(IModel):
    def __init__(self, seed):
        super(mlp_reg, self).__init__()
        self.model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True, verbose=True, random_state=seed) # max_iter is the number of epochs
        self.ACTs = {} 

    def forward(self, images):
        self.ACTs = {'mlp_regression': images}
        return self.model.predict(images)

    def fit(self, x, y):
        """ Fit the regression model using the provided method """
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        self.model.fit(x_scaled, y)
        return self.model
    
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            return self(X)
    
    def get_layers(self, driven):
        return [('mlp_regression', None)]
    
    def get_activations(self, layer):
        print("Returning activations:", self.ACTs.keys())
        return self.ACTs

    def register_hook(self, hook_name, driven):
        print(f"Linear model does not support hooks. Hook name: {hook_name}")
        return []
