
from matplotlib import pyplot as plt
from .data_setup import download_data
import torchinfo
import itertools

"""
File containing various utility functions for PyTorch model training.
"""

from pathlib import Path
from torch import nn, save

def save_model(model:nn.Module, directory:str='.', filename:str='model.pt'):
    """Saves a PyTorch model to a target directory

    Args:
        model: A target PyTorch model to save.
        directory: A directory for saving the model to.
        filename: A filename for the saved model. Should include either ".ppth" or ".pt" as the file extension.

    Example usage:
        save_model(
            model=model_0,
            directory="models",
            filename="my_model.pt"
    """

    assert filename.endswith('.pt') or filename.endswith('.pth'), \
        "filename should end with '.pt' or '.pth"

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_path = directory / filename

    print(f"Saving model into '{save_path}'")
    save(obj=model.state_dict(), f=save_path)

def plot_progress(progress: dict):
    fig, ax = plt.subplots(1,2)
    fig.set_figwidth(15)

    x_b = progress['batch']['x']
    x_t = progress['train']['x']
    x_v = progress['val']['x']

    ax[0].plot(x_b, progress['batch']['loss'], color='#704c70',  label='batch')
    ax[0].plot(x_t, progress['train']['loss'],
               markerfacecolor='none', markeredgecolor='#179ac2',
               linestyle='dashed', marker='o',  label='train')
    ax[0].plot(x_v, progress['val']['loss'],
               markerfacecolor='none', markeredgecolor='#735773',
               linestyle='dashed', marker='o',  label='val')
    ax[0].set_title('Loss')
    ax[0].legend()

    ax[1].plot(x_b, progress['batch']['acc'], color='#59704c', label='batch')
    ax[1].plot(x_t, progress['train']['acc'],
               markerfacecolor='none', markeredgecolor='#179ac2',
               linestyle='dashed', marker='o',  label='train')
    ax[1].plot(x_v, progress['val']['acc'],
               markerfacecolor='none', markeredgecolor='#735773',
               linestyle='dashed', marker='o',  label='val')
    ax[1].set_title('Accuracy')
    ax[1].legend()


def summary(input_size):

    def method(self):
        print(torchinfo.summary(
            self,
            input_size=input_size,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["depth"]
        ))
    return method


def prepare_experiments(search_space):
    keys = search_space.keys()
    all_combinations = itertools.product(*search_space.values())

    experiments_as_dicts = [
        dict(zip(keys, combination)) for combination in all_combinations
    ]

    return experiments_as_dicts


class ExpWrapper:

    def __init__(self, obj, expression):
        self.obj = obj
        self.exp = expression

    def __str__(self):
        return self.exp

    def __repr__(self):
        return self.__str__()

    def extract(self):
        return self.obj


def two_datasets():
    urls = [
        'https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip',
        'https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip'
    ]

    names = ['PSS_0.1', 'PSS_0.2']

    data_paths = []
    for name, url in zip(names, urls):
        download_data('data', name, url)
        data_paths.append(ExpWrapper(Path('data') / name, name))
    return data_paths

