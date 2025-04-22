import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.calibration import LabelEncoder

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
        train_dataset = ITDataSet(train_data[0], train_data[1])
        val_dataset = ITDataSet(val_data[0], val_data[1])
        num_classes = len(set(train_data[1]))
    elif driven == 'data':
        train_dataset = TensorDataset(
            torch.from_numpy(train_data[0]).float(),
            torch.from_numpy(train_data[2]).float()
        )

        val_dataset = TensorDataset(
            torch.from_numpy(val_data[0]).float(),
            torch.from_numpy(val_data[2]).float()
        )
        num_classes = 1 # only one output for regression
    else:
        raise ValueError("driven must be either 'task' or 'data'")
    
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return trainloader, valloader, num_classes

class ITDataSet(Dataset):
    def __init__(self, images, labels):
        super().__init__()
        self.images = torch.from_numpy(images).float()
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        self.labels = encoded_labels

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label