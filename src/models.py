"""Contains PyTorch model code to instantiate a TinyVGG model from the CNN
 Explainer Website"""

from torch import nn
from torchinfo import summary
from . import utils
from types import MethodType
import torch


class TinyVGG(nn.Module):

    """Creates the TinyVGG architecture.

      Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
      See the original architecture here: https://poloclub.github.io/cnn-explainer/

      Args:
          input_shape
    """

    def __init__(self,
                 input_shape: list | tuple,
                 classes: list | tuple,
                 device: str='cpu'):
        super().__init__()

        self.input_shape = input_shape
        self.classes = classes
        self.device = device

        arc = {
            'f': (64,  64,  128, 256),
            'k': (5,   3,   2,   2),
            'p': (2,   1,   1,   0),
            'l': (0,   1,   0,   1)
        }

        n_blocks = len(arc['f'])

        assert len(input_shape) == 3

        in_channels, w, h = input_shape

        self.layers = nn.Sequential()
        for i in range(n_blocks):
            self.layers.add_module(f'conv_{i}',
                                   nn.Conv2d(
                                       in_channels=in_channels,
                                       out_channels=arc['f'][i],
                                       kernel_size=arc['k'][i],
                                       padding=arc['p'][i],
                                       stride=1,
                                       device=device
                                   ))
            self.layers.add_module(f'actv_{i}', nn.ReLU())
            self.layers.add_module(f'pool_{i}', nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )) if arc['l'][i] else None

            in_channels = arc['f'][i]

        dummy_x = torch.ones([1, *input_shape], device=device).float()
        n = self(dummy_x).shape.numel()

        self.layers.add_module('fc', nn.Flatten())
        self.layers.add_module('classification', nn.Linear(in_features=n, out_features=len(classes), device=device))

    def forward(self, x):
        return self.layers(x)

    def predict(self, x, label=True):
        self.eval()
        with torch.inference_mode():
            i = torch.argmax(nn.functional.softmax(self.layers(x.to(self.device)), dim=1), dim=1)
            return [self.classes[i] for i in i] if label else i

    def summary(self):
        summary(self, input_size=[1, *self.input_shape])


def _check_content(block, attr):
    try:
        for sub in block:
            if hasattr(sub, attr):
                return sub.__getattribute__(attr)
            else:
                _check_content(block, attr)
    except TypeError:
        pass

    return None


def fine_tune_on_classes(pretrained_model: nn.Module,
                         target: list,
                         dropout: float = 0.2,
                         device: str = 'cpu',
                         input_size: tuple[int, int] = (224, 224)
                         ):
    """
    Adapts a pre-trained model to a new target

    Args:
        pretrained_model:
            nn.Module model with pretrained weights
        target:
            list of class names representing the classification target
        dropout:
            Percentage for dropout probability at the output layer
        device:
            Name of device where data tensors are allocated.
        input_size:
            Width and height of input to be displayed in the model's summary

    Returns:
        nn.Module for a pretrained model modified to the given target.
    """

    assert hasattr(pretrained_model, 'features'), \
        "Expected a 'features' block in the provided model"
    assert hasattr(pretrained_model, 'classifier'), \
        "Expected a 'classifier' block in the provided model"

    frozen_block = pretrained_model.features
    for param in frozen_block.parameters():
        param.requires_grad = False

    out_channels = None
    for sub in frozen_block[-1:]:
        out_channels = _check_content(sub, 'out_channels')
        if out_channels:
            break
    if not out_channels:
        raise AttributeError(f"Could not find a layer with 'out_channels' in the given model")

    pretrained_model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=out_channels, out_features=len(target))
    )

    fun = MethodType(utils.summary(input_size=[1, 3, *input_size]), pretrained_model)
    pretrained_model.summary = fun

    return pretrained_model.to(device)
