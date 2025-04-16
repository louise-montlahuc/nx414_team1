import argparse

import torch
from torch import nn

from Week_8.models import build
from Week_8.tools.optimizer import make_optimizer
from Week_8.tools.trainer import do_train

def main(args):
    trainloader, testloader = make_dataloader()
    model = build(args.name, args.pretrained)
    optim = make_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    do_train(model, trainloader, testloader, optim, criterion, args.epochs, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default='resnet18', help='Model name', type=str)
    parser.add_argument('-p', '--pretrained', action='store_true', help='Whether to load a pretrained model')
    parser.add_argument('-o', '--optimizer', default='adamw', help='Optimizer name', type=str)
    parser.add_argument('--lr', default=1e-3, help='Learning rate', type=float)
    parser.add_argument('--weight_decay', default=1e-4, help='Weight decay factor', type=float)
    parser.add_argument('-b', '--batch_size', default=32, help='Size of batches', type=int)
    args = parser.parse_args()
    main(args)