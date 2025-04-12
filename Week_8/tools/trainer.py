import torch
from tqdm import tqdm

def do_train(model, train_loader, val_loader, optim, scheduler, criterion, epochs, device):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.
        train_mae = 0.

        for data, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} - Training'):
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            optim.zero_grad()
            out = model(data, dataset_names)
            loss = criterion(out, targets)

            train_mae += torch.mean(torch.abs(out - targets)).item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optim.step()
            scheduler.step()

            train_loss += loss.item() * data.size(0)

        train_loss /= len(train_loader.dataset)
        train_mae /= len(train_loader.dataset)
       
        if val_loader is not None:
            model.eval()
            valid_loss = 0.
            valid_mae = 0.

            with torch.no_grad():
                for data, targets, dataset_names in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} - Validation'):
                    data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                    out = model(data, dataset_names)

                    loss = criterion(out, targets)
                    valid_loss += loss.item() * data.size(0)

                    valid_mae += torch.mean(torch.abs(out - targets)).item()

            valid_loss /= len(val_loader.dataset)
            valid_mae /= len(val_loader.dataset)

            # TODO save model if best valid mae