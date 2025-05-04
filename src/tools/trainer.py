import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import r2_score

from tools.dataloader import make_dataloader
from tools.optimizer import make_optimizer
from tools.scheduler import make_scheduler
from utils.plotter import Plotter

def finetune(model, args):
    trainloader, valloader, num_classes = make_dataloader(16, args.driven)
    # Cut and set head after specific layer
    model = model.change_head(args.layer, num_classes)
    print(model)
    optim = make_optimizer(args.optimizer, model, args.lr, 1e-5)
    scheduler = make_scheduler(optim, 4, args, trainloader)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)

    if args.driven == 'task':
        # Finetunes the model by object (image) classification
        criterion = nn.CrossEntropyLoss()
        do_train(model, trainloader, valloader, optim, criterion, scheduler, args.epochs, device, args)
    elif args.driven == 'data':
        # Finetunes the model by data (neural) regression
        criterion = nn.MSELoss()
        do_train(model, trainloader, valloader, optim, criterion, scheduler, args.epochs, device, args)
    else:
        raise ValueError(f"Unknown driven argument: {args.driven}. Supported methods are 'task' and 'data'.")
    
    return model

def do_train(model, train_loader, val_loader, optim, criterion, scheduler, epochs, device, args):
    train_losses = []
    valid_losses = []
    lrs = []
    r2s = []

    best_valid_loss = float('inf')
    for epoch in range(epochs):
        # ---- Training ----
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
        print(f'Train Loss: {train_loss:.4f}')
       
        # ---- Validation ----
        model.eval()
        valid_loss = 0.
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, targets in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} - Validation'):
                data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                out = model(data)

                loss = criterion(out, targets)
                valid_loss += loss.item() * data.size(0)

                all_preds.append(out.cpu())
                all_targets.append(targets.cpu())

        valid_loss /= len(val_loader.dataset)

        # Stack all batches to compute R2
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        r2 = r2_score(all_targets, all_preds)
        r2s.append(r2)

        print(f'Validation Loss: {valid_loss:.4f} | R^2 score: {r2:.4f}')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'saved/models/{args.name}_best_model.pth')
            print(f'Saved best model with validation loss: {best_valid_loss:.4f}')

        # ---- Logging ----
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        lrs.append(scheduler.get_last_lr()[0])
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.10f}')

    # Plot the training history
    Plotter.plot_training_history(train_losses, valid_losses, r2s, lrs, epochs, args.name, args.layer)