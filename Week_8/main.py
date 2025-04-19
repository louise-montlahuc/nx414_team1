import argparse

import torch
from torch import nn
from scipy.stats import pearsonr

from models.IModel import IModel
from models.build import make_model
from tools.dataloader import make_dataloader, get_data
from tools.optimizer import make_optimizer
from tools.trainer import do_train
from tools.regression import fit
from utils.plotter import Plotter
from utils.utils import download_data

def main(args):
    download_data('./data') # download the data if not already done
    model = make_model(args.name)
    print(model)
    device = torch.device("cpu")
    if args.finetune: # don't finetune for the moment
        if isinstance(model, IModel) and torch.cuda.is_available():
            device = torch.device("cuda")
            model = model.to(device)
        optim = make_optimizer(model, args.optimizer, args.lr, args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        trainloader, testloader = make_dataloader(args.batch_size, args.driven)
        do_train(model, trainloader, testloader, optim, criterion, args.epochs, device)

    if isinstance(model, IModel):
        linprob(model, device, args.name)

def linprob(model, device, name):
    # Perform linear probing
    model.eval()
    layer_regressions = dict()
    with torch.no_grad():
        train_data, val_data = get_data()
        stimulus_train, _, spikes_train = train_data
        stimulus_train = torch.from_numpy(stimulus_train).to(device)
        spikes_train = torch.from_numpy(spikes_train).to(device)
        stimulus_val, _, spikes_val = val_data
        stimulus_val = torch.from_numpy(stimulus_val).to(device)
        spikes_val = torch.from_numpy(spikes_val).to(device)
        print('Registering hooks...')
        handles = model.register_hook(args.hook)
        # Fit the regression on the activations of the training set
        print('Computing activations...')
        model(stimulus_train)
        activations = model.get_activations(args.hook)
        print('Fitting the regressions...')
        for layer_name, _ in model.get_layers():
            print('\tLayer:', layer_name)
            layer_regressions[layer_name] = fit(activations[layer_name], spikes_train, method='linear')

        # Test the regression on the validation set
        print('Testing the regression...')
        model.reset_activations()
        model(stimulus_val)

        ## Remove handles
        for handle in handles:
            handle.remove()

        activations = model.get_activations(args.hook)
        for layer_name, regr in layer_regressions.items():
            print('\tLayer:', layer_name)
            pred_activity = regr.predict(activations[layer_name])
            correlations = []
            for i in range(spikes_val.shape[1]):
                corr, _ = pearsonr(pred_activity[:, i], spikes_val[:, i])
                correlations.append(corr)
            Plotter.save_corr_plot(
                data=correlations,
                title=f'[{name}] Correlation between predicted and actual spikes for layer {layer_name}',
                path=f'saved/{name}_correlation_layer_{layer_name}.png'
            )
        print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default='ResNet18', help='Model name', type=str)
    parser.add_argument('-d', '--driven', default='task', help='Task- or data-driven model', type=str)
    parser.add_argument('--hook', default='all', help='Hook name', type=str)
    parser.add_argument('-o', '--optimizer', default='adamw', help='Optimizer name', type=str)
    parser.add_argument('--lr', default=1e-3, help='Learning rate', type=float)
    parser.add_argument('--weight_decay', default=1e-4, help='Weight decay factor', type=float)
    parser.add_argument('-b', '--batch_size', default=32, help='Size of batches', type=int)
    parser.add_argument('-e', '--epochs', default=10, help='Number of epochs', type=int)
    parser.add_argument('-f', '--finetune', action='store_true', help='Whether to finetune the model')
    args = parser.parse_args()
    main(args)