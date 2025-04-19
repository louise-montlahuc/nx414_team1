import os

import torch
from scipy.stats import pearsonr

from tools.dataloader import get_data
from tools.regression import fit
from utils.plotter import Plotter

def linprob(model, args):
    """Linear probing of the model.

    Args:
        model (IModel): the model.
        args (argparse.Namespace): the arguments passed to the script.
    """
    model.eval()
    layer_regressions = dict()

    with torch.no_grad():
        train_data, val_data = get_data()
        stimulus_train, _, spikes_train = train_data
        stimulus_train = torch.from_numpy(stimulus_train)
        spikes_train = torch.from_numpy(spikes_train)
        stimulus_val, _, spikes_val = val_data
        stimulus_val = torch.from_numpy(stimulus_val)
        spikes_val = torch.from_numpy(spikes_val)
        print('Registering hooks...')
        handles = model.register_hook(args.hook)
        # Fit the regression on the activations of the training set
        print('Computing activations...')
        if args.saved and os.path.exists(f'./saved/activations/{args.name}_{args.hook}_train_activations.pt'):
            print('Loading saved training activations...')
            activations = torch.load(f'./saved/activations/{args.name}_{args.hook}_train_activations.pt', weights_only=False)
        else:
            model(stimulus_train)
            activations = model.get_activations(args.hook)
            torch.save(activations, f'./saved/activations/{args.name}_{args.hook}_train_activations.pt')

        print('Fitting the regressions...')
        for layer_name, _ in model.get_layers():
            print('\tLayer:', layer_name)
            layer_regressions[layer_name] = fit(activations[layer_name], spikes_train, method=args.probing)

        # Test the regression on the validation set
        print('Testing the regression...')
        model.reset_activations()

        if args.saved and os.path.exists(f'./saved/activations/{args.name}_{args.hook}_valid_activations.pt'):
            print('Loading saved validation activations...')
            activations = torch.load(f'./saved/activations/{args.name}_{args.hook}_valid_activations.pt', weights_only=False)
        else:
            model(stimulus_val)
            activations = model.get_activations(args.hook)
            torch.save(activations, f'./saved/activations/{args.name}_{args.hook}_valid_activations.pt')

        ## Remove handles
        for handle in handles:
            handle.remove()
            
        for layer_name, regr in layer_regressions.items():
            print('\tLayer:', layer_name)
            if isinstance(regr, tuple): # MLP returns both the model and the scaler
                regr, scaler = regr
                activations[layer_name] = scaler.transform(activations[layer_name])
            pred_activity = regr.predict(activations[layer_name])
            correlations = []
            for i in range(spikes_val.shape[1]):
                corr, _ = pearsonr(pred_activity[:, i], spikes_val[:, i])
                correlations.append(corr)
            Plotter.save_corr_plot(
                data=correlations,
                title=f'[{args.name}/{args.hook}/{args.probing}]\nCorrelation between predicted and actual spikes for layer {layer_name}',
                path=f'./saved/{args.name}_{args.hook}_{args.probing}_correlation_layer_{layer_name}.png'
            )
        print('Done!')