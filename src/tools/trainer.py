import torch
from torch import nn
from tqdm import tqdm

from tools.dataloader import make_dataloader
from tools.optimizer import make_optimizer

def finetune(model, device, args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)

    trainloader, valloader = make_dataloader(args.batch_size, args.driven)
    optim = make_optimizer(model, args.optimizer, args.lr, args.weight_decay)

    if args.driven == 'task':
        # Finetunes the model on doing object (image) classification
        criterion = nn.CrossEntropyLoss()
        do_train(model, trainloader, valloader, optim, criterion, args.epochs, device)
    elif args.driven == 'data':
        raise NotImplementedError("Finetuning on neural data is not implemented yet.")
    else:
        raise ValueError(f"Unknown driven argument: {args.driven}. Supported methods are 'task' and 'data'.")

def do_train(model, train_loader, val_loader, optim, scheduler, criterion, epochs, device):
    best_valid_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.

        for data, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} - Training'):
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            optim.zero_grad()
            out = model(data)
            loss = criterion(out, targets)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optim.step()
            scheduler.step()

            train_loss += loss.item() * data.size(0)

        train_loss /= len(train_loader.dataset)
       
        if val_loader is not None:
            model.eval()
            valid_loss = 0.

            with torch.no_grad():
                for data, targets in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} - Validation'):
                    data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                    out = model(data)

                    loss = criterion(out, targets)
                    valid_loss += loss.item() * data.size(0)

            valid_loss /= len(val_loader.dataset)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'saved/best_model.pth')
                print(f'Saved best model with validation loss: {best_valid_loss:.4f}')
            