import torch

def make_optimizer(name, model, lr, weight_decay):
    classif_params = list(model.fc.parameters())
    all_params = list(model.parameters())
    classif_param_ids = set(id(p) for p in classif_params)
    base_params = [p for p in all_params if id(p) not in classif_param_ids]

    if name == 'adamw':
        return torch.optim.AdamW([
            {'params': base_params, 'lr': 0.1 * lr},
            {'params': classif_params, 'lr': lr}
        ], weight_decay=weight_decay)
    elif name == 'sgd':
        return torch.optim.SGD([
            {'params': base_params, 'lr': 0.1 * lr},
            {'params': classif_params, 'lr': lr}
        ], weight_decay=weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {name} is not implemented.")  