import os
import torch
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from tools.dataloader import get_data
from tools.regression import fit
from utils.plotter import Plotter

from models.linear_reg import linear_reg
from models.ridge_reg import ridge_reg
from models.mlp_reg import mlp_reg

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

        # If linear regression model, ensure it is fitted 
        if isinstance(model, (linear_reg, ridge_reg, mlp_reg)):
            print('Fitting the linear regression model...')
            n_stimulus = stimulus_train.shape[0]
            stimulus_train = stimulus_train.reshape(n_stimulus, -1)
            model.fit(stimulus_train, spikes_train)  

            n_stimulus_val = stimulus_val.shape[0]
            stimulus_val = stimulus_val.reshape(n_stimulus_val, -1)
        
        # Fit the regression on the activations of the training set
        print('Computing activations...')
        save_folder = os.path.join(os.getcwd(), 'saved')
        if args.saved and not args.finetune and os.path.exists(f'{save_folder}/activations/{args.name}_{args.hook}_train_activations.pt'):
            print('Loading saved training activations...')
            activations = torch.load(f'{save_folder}/activations/{args.name}_{args.hook}_train_activations.pt', weights_only=False)
        else:
            model(stimulus_train)
            activations = model.get_activations(args.hook)
            torch.save(activations, f'{save_folder}/activations/{args.name}_{args.hook}_train_activations.pt')

        print('Fitting the regressions...')
        for layer_name, _ in model.get_layers():
            print('\tLayer:', layer_name)
            layer_regressions[layer_name] = fit(activations[layer_name], spikes_train, method=args.probing)

        # Test the regression on the validation set
        print('Testing the regression...')
        model.reset_activations()

        if args.saved and not args.finetune and os.path.exists(f'{save_folder}/activations/{args.name}_{args.hook}_valid_activations.pt'):
            print('Loading saved validation activations...')
            activations = torch.load(f'{save_folder}/activations/{args.name}_{args.hook}_valid_activations.pt', weights_only=False)
        else:
            model(stimulus_val)
            activations = model.get_activations(args.hook)
            torch.save(activations, f'{save_folder}/activations/{args.name}_{args.hook}_valid_activations.pt')

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

            # Compute mean R² score
            r2 = r2_score(spikes_val, pred_activity)
            name = f'{args.name}_{layer_name}' if not args.finetune else f'{args.name}_finetuned_{layer_name}'
            new_score = {name: r2}
            Plotter.update_r2_score_csv(new_score, f"{save_folder}/r2_scores.csv")
            Plotter.save_r2_table(
                path_csv=f"{save_folder}/r2_scores.csv",
                path_png=f"{save_folder}/r2_scores.png"
            )

            Plotter.save_corr_plot(
                data=correlations,
                title=f'[{args.name}/{args.hook}/{args.probing}]\nCorrelation between predicted and actual spikes for layer {layer_name}',
                path=f'{save_folder}/figures/{args.name}{"_finetuned" if args.finetune else ""}_{args.hook}_{args.probing}_correlation_{layer_name}.png'
            )
        print('Done!')