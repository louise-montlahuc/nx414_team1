import os
import torch
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

from tools.dataloader import get_data
from tools.regression import fit
from utils.plotter import Plotter

from models.linear_reg import linear_reg
from models.ridge_reg import ridge_reg
from models.mlp_reg import mlp_reg

def score(model, layer_name, args):
    with torch.no_grad():
        _, val_data = get_data()
        stimulus_val, _, spikes_val = val_data
        stimulus_val = torch.from_numpy(stimulus_val)
        spikes_val = torch.from_numpy(spikes_val)

        if args.pca:
            pca = PCA(n_components=1000)
            stimulus_val = pca.transform(stimulus_val)

        print('Computing activations...')
        out = model(stimulus_val)

        _compute_score(spikes_val, out, layer_name, args)
        print('DONE!')

def linprob(model, seed, args):
    """Linear probing of the model.

    Args:
        model (IModel): the model.
        args (argparse.Namespace): the arguments passed to the script.
    """
    if args.driven == 'data' and args.finetune:
        _linprob_finetuned(model, seed, args)
    else:
        _linprob(model, seed, args)

def _linprob_finetuned(model, seed, args):
    """
    In this case, the finetuning is done after a given layer (in args). We thus need to probe
    only the previous to last layer (last is MLP).
    """
    model.eval()
    with torch.no_grad():
        train_data, val_data = get_data()
        stimulus_train, _, spikes_train = train_data
        stimulus_train = torch.from_numpy(stimulus_train)
        spikes_train = torch.from_numpy(spikes_train)
        stimulus_val, _, spikes_val = val_data
        stimulus_val = torch.from_numpy(stimulus_val)
        spikes_val = torch.from_numpy(spikes_val)

        if args.pca:
            pca = PCA(n_components=1000)
            stimulus_train_flat = stimulus_train.reshape(stimulus_train.shape[0], -1)
            stimulus_val_flat = stimulus_val.reshape(stimulus_val.shape[0], -1)
            stimulus_train_pca = pca.fit_transform(stimulus_train_flat)
            stimulus_val_pca = pca.transform(stimulus_val_flat)
            stimulus_train = stimulus_train_pca.reshape(stimulus_train.shape)
            stimulus_val = stimulus_val_pca.reshape(stimulus_val.shape)

        print('Registering hooks...')
        handles = model.register_hook(args.hook, args.driven)
        
        print('Computing activations...')
        model(stimulus_train)
        activations = model.get_activations(args.hook)

        print('\tLayer:', args.layer)
        regression = fit(activations, spikes_train, method=args.probing, seed=seed)

        # Test the regression on the validation set
        print('Testing the regression...')
        model.reset_activations()

        model(stimulus_val)
        activations = model.get_activations(args.hook)

        ## Remove handles
        for handle in handles:
            handle.remove()
        
        print('\tLayer:', args.layer)
        if isinstance(regression, tuple): # MLP returns both the model and the scaler
            regression, scaler = regression
            activations = scaler.transform(activations)
        pred_activity = regression.predict(activations)
        _compute_score(spikes_val, pred_activity, args.layer, args)
        print('Done!')

def _linprob(model, seed, args):
    model.eval()
    layer_regressions = dict()
    save_folder = os.path.join(os.getcwd(), 'saved')

    with torch.no_grad():
        train_data, val_data = get_data()
        stimulus_train, _, spikes_train = train_data
        stimulus_train = torch.from_numpy(stimulus_train)
        spikes_train = torch.from_numpy(spikes_train)
        stimulus_val, _, spikes_val = val_data
        stimulus_val = torch.from_numpy(stimulus_val)
        spikes_val = torch.from_numpy(spikes_val)

        if args.pca:
            pca = PCA(n_components=1000)
            stimulus_train_flat = stimulus_train.reshape(stimulus_train.shape[0], -1)
            stimulus_val_flat = stimulus_val.reshape(stimulus_val.shape[0], -1)
            stimulus_train_pca = pca.fit_transform(stimulus_train_flat)
            stimulus_val_pca = pca.transform(stimulus_val_flat)
            #stimulus_train = stimulus_train_pca.reshape(stimulus_train.shape)
            #stimulus_val = stimulus_val_pca.reshape(stimulus_val.shape)

        if isinstance(model, (linear_reg, ridge_reg, mlp_reg)):
            handles = []
            # Reshaping data
            print('Fitting the linear regression model...')
            n_stimulus = stimulus_train.shape[0]
            stimulus_train = stimulus_train.reshape(n_stimulus, -1) 

            n_stimulus_val = stimulus_val.shape[0]
            stimulus_val = stimulus_val.reshape(n_stimulus_val, -1)
        else:
            print('Registering hooks...')
            handles = model.register_hook(args.hook, args.driven)
        
        if isinstance(model, (linear_reg, ridge_reg, mlp_reg)):
            model.fit(stimulus_train, spikes_train)  
            model(stimulus_train)
            activations = model.get_activations(args.hook)
        else:
            # Fit the regression on the activations of the training set
            print('Computing activations...')
            save_folder = os.path.join(os.getcwd(), 'saved')
            if args.saved and os.path.exists(f'{save_folder}/activations/{args.name}_{args.hook}{"_pca" if args.pca else ""}_train_activations.pt'):
                print('Loading saved training activations...')
                activations = torch.load(f'{save_folder}/activations/{args.name}_{args.hook}{"_pca" if args.pca else ""}_train_activations.pt', weights_only=False)
            else:
                model(stimulus_train)
                activations = model.get_activations(args.hook)
                torch.save(activations, f'{save_folder}/activations/{args.name}_{args.hook}{"_pca" if args.pca else ""}_train_activations.pt')

        print('Fitting the regressions...')
        if isinstance(model, (linear_reg, ridge_reg, mlp_reg)):
            for layer_name, _ in model.get_layers(args.driven):
                layer_regressions[layer_name] = model
        else:
            for layer_name, _ in model.get_layers(args.driven):
                print('\tLayer:', layer_name)
                layer_regressions[layer_name] = fit(activations[layer_name], spikes_train, method=args.probing, seed=seed)

        # Test the regression on the validation set
        print('Testing the regression...')
        model.reset_activations()

        if args.saved and os.path.exists(f'{save_folder}/activations/{args.name}_{args.hook}{"_pca" if args.pca else ""}_valid_activations.pt'):
            print('Loading saved validation activations...')
            activations = torch.load(f'{save_folder}/activations/{args.name}_{args.hook}{"_pca" if args.pca else ""}_valid_activations.pt', weights_only=False)
        else:
            model(stimulus_val)
            activations = model.get_activations(args.hook)
            torch.save(activations, f'{save_folder}/activations/{args.name}_{args.hook}{"_pca" if args.pca else ""}_valid_activations.pt')

        ## Remove handles
        for handle in handles:
            handle.remove()
        
        for layer_name, regr in layer_regressions.items():
            print('\tLayer:', layer_name)
            if isinstance(regr, tuple): # MLP returns both the model and the scaler
                regr, scaler = regr
                activations[layer_name] = scaler.transform(activations[layer_name])
            pred_activity = regr.predict(activations[layer_name])
            _compute_score(spikes_val, pred_activity, layer_name, args)
        print('Done!')

def _compute_score(y_true, y_pred, layer_name, args):
    save_folder = os.path.join(os.getcwd(), 'saved')
    correlations = []
    for i in range(y_true.shape[1]):
        corr, _ = pearsonr(y_pred[:, i], y_true[:, i])
        correlations.append(corr)

    # Compute mean R² score
    r2 = r2_score(y_true, y_pred)
    name = f'{args.name}_{"finetuned" if args.finetune else "pretrained"}_{layer_name}{f"_{args.probing}" if not args.finetune else ""}_{args.hook}{"_pca" if args.pca else ""}'
    new_score = {name: r2}
    Plotter.update_r2_score_csv(new_score, f"{save_folder}/r2_scores.csv")
    Plotter.save_r2_table(
        path_csv=f"{save_folder}/r2_scores.csv",
        path_png=f"{save_folder}/r2_scores.png"
    )

    Plotter.save_corr_plot(
        data=correlations,
        title=f'[{args.name}/{args.probing if not args.finetune else "finetuned"}/{args.hook}]\nCorrelation between predicted and actual spikes for layer {layer_name}',
        path=f'{save_folder}/figures/corr_{args.name}_{"finetuned" if args.finetune else "pretrained"}_{layer_name}{f"_{args.probing}" if not args.finetune else ""}_{args.hook}_{"pca" if args.pca else ""}.png'
    )