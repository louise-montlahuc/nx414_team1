import tqdm
import torch

def do_test(model, test_loader, criterion, device):
    model.eval()
    loss = 0.0
    mae = 0.0
    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc=f'Testing'):
            data, targets = data.to(device), targets.to(device)
            out = model(data)
            loss = criterion(out, targets)

            loss += loss.item()
            mae += torch.mean(torch.abs(out - targets)).item()

        loss /= len(test_loader)
        mae /= len(test_loader)

    return loss, mae