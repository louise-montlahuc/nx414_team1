from utils import load_it_data

def make_dataloader():
    stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val = load_it_data('./')
    return (stimulus_train, spikes_train), (stimulus_val, spikes_val)