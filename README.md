# Team 1 - Brain-like computation and intelligence (NX-414)
Members: Kolly Florian, Mikami Sarah, Montlahuc Louise
## Project description
The objectives of the project are:
- Predict neural activity using linear regression from images and from neural network layers.
- Quantify the goodness of the model
- Compare the results across the network layers and between trained/random neural network
- Predict the neural activity using a neural network in a data-driven approach
- Develop the most accurate model for predicting IT neural activity

Specifically, we use the data from the following [paper](https://www.jneurosci.org/content/jneuro/35/39/13402.full.pdf). The behavioral experiment consisted in showing to non-human primates some images while recording the neural activity with multielectrode arrays from the inferior temporal (IT) cortex. In the data we provided you, the neural activity and the images are already pre-processed and you will have available the images and the corresponding average firing rate (between 70 and 170 ms) per each neuron.

## Mini project 1: Predicting neural activity

Deadline: May 7, 2025 </br>
Part 1: Predict the neural activity from pixels </br>
Part 2: Predict the neural activity with the task-driven modeling approach </br>
&rarr; In `week 6` </br>
Part 3: Predict the neural activity using a data-driven approach </br>
&rarr; In `week 7` </br>
Part 4: Find the best model </br>
&rarr; In `week 8`, using previous findings </br>

## Setup the environment
TODO

## Folder structure
- src
    - data: folder where the data will be downloaded
    - models: contains the models
    - saved: folder containing the different files that we save during usage
        - activations: contains the activations of the models for specific 1) model, 2) hook, 3) step (train or valid), 4) layer
    - tools
        - dataloader.py: handles the data and PyTorch dataloaders
        - optimizer.py: handles the optimizers
        - prober.py: handles the different types of probing
        - regression.py: handles the different types of regression
        - trainer.py: handles the training and finetuning of the models
    - utils: various utility functions (visualization, loading data, etc.)
    - main.py: main file to run the code
- Week_6: notebook for the first week
- Week_7: notebook for the second week
- Week_8: nothing to see here, just a placeholder

## Commands
The code should be run using the command line. The main file is `main.py` and the arguments are as follows:
```bash
python main.py -n <model_name> -d <task/data> -o <optimizer_name> -k <all/pca> -f <finetune> --lr <lr> -p <probing_name>
```

- `-n` or `--name`: name of the model to use. The default is `ResNet18`.
    Models available:
    - Simple
        - TODO
    - Pretrained
        - `ResNet18`
        - `ResNet50`
        - `ResNeXt`
        - `ConvNeXt`
        - `ViT`
        - `DinoV2`
    
    For the pretrained models, the layers to test are automatically selected.
- `-d` or `--driven`: type of task to use. The default is `task`.
    - `task`: task-driven modeling approach
    - `data`: data-driven modeling approach
- `-o` or `--optimizer`: type of optimizer to use. The default is `adam`.
    - `adam`: Adam optimizer
    - `sgd`: Stochastic Gradient Descent
- `-k` or `--hook`: how the activations are saved via the hook. The default is `all`.
    - `all`: all the activations
    - `pca`: PCA of the activations
- `-p` or `--probing`: type of probing to use. The default is `linear`.
    - `linear`: linear regression probing
    - `ridge`: Ridge regression probing
    - `mlp`: MLP regression probing
- `-f` or `--finetune`: whether to finetune the model or not. The default is `False`.
    Note: a finetuned model will never use the saved activations.
- `--lr`: learning rate. The default is `1e-3`.
- `--epochs`: number of epochs to train the model. The default is `10`.
- `--saved`: whether to use the saved activations or not. The default is `False`.