import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.utils import load_it_data

def get_data():
    """Get the data from the IT dataset.

    Returns:
        tuple: tuples (stimulus, objects, spikes) for training and validation sets.
    """
    stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val = load_it_data('./data/')
    return (stimulus_train, objects_train, spikes_train), (stimulus_val, objects_val, spikes_val)

def make_dataloader(batch_size, driven='task'):
    """Create a PyTorch data loader, depending on whether the model is data-driven or
    task-driven.

    Args:
        batch_size (int): Batch size for the data loader.
        driven (str, optional): 'data' if the model is data-driven, 'task' if the model is
        task-driven. Defaults to 'task'.
    """
    train_data, val_data = get_data()
    if driven == 'task':
        train_dataset = TensorDataset(
            torch.from_numpy(train_data[0]).float(),
            torch.from_numpy(train_data[2]).float()
        )

        val_dataset = TensorDataset(
            torch.from_numpy(val_data[0]).float(),
            torch.from_numpy(val_data[2]).float()
        )
    elif driven == 'data':
        train_dataset = TensorDataset(
            torch.from_numpy(train_data[0]).float(),
            torch.from_numpy(train_data[1]).float()
        )

        val_dataset = TensorDataset(
            torch.from_numpy(val_data[0]).float(),
            torch.from_numpy(val_data[1]).float()
        )
    else:
        raise ValueError("driven must be either 'task' or 'data'")
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return trainloader, valloader