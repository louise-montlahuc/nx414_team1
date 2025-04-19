import torch

def make_optimizer(name, model, lr, weight_decay):
    if name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {name} is not implemented.")
    