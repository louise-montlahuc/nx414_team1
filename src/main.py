import argparse

from utils.utils import download_data
from models.IModel import IModel
from models.build import make_model
from tools.trainer import finetune
from tools.prober import linprob

def main(args):
    download_data('./data/') # download the data if not already done
    model = make_model(args.name)
    print(model)
    if args.finetune:
        finetune(model, args)

    if isinstance(model, IModel):
        linprob(model, args)
    else:
        raise NotImplementedError("Louise") # TODO

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default='ResNet18', help='Model name', type=str)
    parser.add_argument('-d', '--driven', default='task', help='Task- or data-driven model', type=str)
    parser.add_argument('-k', '--hook', default='all', help='Hook name', type=str)
    parser.add_argument('-o', '--optimizer', default='adamw', help='Optimizer name', type=str)
    parser.add_argument('-f', '--finetune', action='store_true', help='Whether to finetune the model')
    parser.add_argument('-p', '--probing', default='linear', help='Probing method', type=str)
    parser.add_argument('--nosaved', action='store_false', dest='saved', help='Do not use saved activations, if any. Default is to use them')
    args = parser.parse_args()
    main(args)