import argparse

import torch

from utils.seed import set_seed
from utils.utils import download_data
from models.build import make_model
from tools.trainer import finetune
from tools.prober import linprob, score

def main(args):
    seed = set_seed()
    download_data('./data/') # Download the data if not already done
    model = make_model(args.name, seed)
    if args.finetune:
        model = finetune(model, args)
        model.load_state_dict(torch.load(f'./saved/models/{args.name}_best_model.pth'))
        model.to('cpu') # Make sure the model is back on CPU
    else:
        print(model)

    if not args.finetune or args.driven == 'task':
        linprob(model, seed, args)
    else:
        # No need to do probing, just pass through the model
        score(model, args.layer, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default='ResNet18', help='Model name', type=str)
    parser.add_argument('-f', '--finetune', action='store_true', help='Whether to finetune the model or use as is')
    parser.add_argument('-d', '--driven', default='data', help='Data- or task-driven finetuning', type=str)
    parser.add_argument('-k', '--hook', default='all', help='Hook name', type=str)
    parser.add_argument('-p', '--probing', default='linear', help='Probing method (only for task-driven)', type=str)
    parser.add_argument('-l', '--layer', default='avgpool', help='Layer name (only for finetuning)', type=str)
    parser.add_argument('-o', '--optimizer', default='adamw', help='Optimizer name', type=str)
    parser.add_argument('--lr', default=1e-5, help='Learning rate for finetuning', type=float)
    parser.add_argument('--epochs', default=10, help='Number of epochs for finetuning', type=int)
    parser.add_argument('--saved', action='store_true', help='Use saved activations, if any. Default is to not use them')
    args = parser.parse_args()
    main(args)