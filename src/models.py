"""Contains PyTorch model code to instantiate a TinyVGG model from the CNN
 Explainer Website"""

from torch import nn
from torchinfo import summary
from torchvision.models.efficientnet import EfficientNet
from . import utils
from types import MethodType
import torch


class PatchEncoder(nn.Module):
    def __init__(self, emb_dim, patch_size, n_patches, dropout=0.0):
        super().__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=emb_dim,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0
            ),                  # (N, C, H, W) -> (N, emb_dim, H // patch_size, W // patch_size)
            nn.Flatten(2),      # (N, emb_dim, H // patch_size, W // patch_size) -> (N, emb_dim, N_patches)
        )

        self.class_token = nn.Parameter(torch.randn(size=(1, 1, emb_dim), requires_grad=True))
        self.position_embeddings = nn.Parameter(torch.randn(size=(1, n_patches+1, emb_dim), requires_grad=True))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        cls_token = self.class_token.expand(x.shape[0], -1, -1)  # (N, 1, emb_dim)

        x = self.patcher(x).permute(0, 2, 1)  # (N, N_patches, emb_dim)
        x = torch.cat([cls_token, x], dim=1)  # (N, N_patches+1, emb_dim)
        x = self.position_embeddings + x  # (N, N_patches+1, emb_dim)
        x = self.dropout(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, heads, dropout=0.0):
        super().__init__()

        self.norm = nn.LayerNorm(normalized_shape=embed_dim)

        # embed_dim will be split across num_heads
        # (i.e. each head will have dimension embed_dim // num_heads).
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=heads, dropout=dropout, batch_first=True)

        # Source and target sequence lengths are both set to N_patches+1,
        # thus: in_features=out_features=embed_dim. The *3 is to process q, k, v in one go.
        self.qkv = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim*3),
            nn.LayerNorm(normalized_shape=embed_dim*3)
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.norm(x)  # (N, N_patches+1, emb_dim)

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        x, _ = self.attention(q, k, v, need_weights=False)
        x = self.dropout(x)
        return x  # (N, N_patches+1, emb_dim)


class TransformerEncoder(nn.Module):

    def __init__(self, embed_dim, heads, mlp_dim, dropout=0.0, head_dropout=0.0):
        super().__init__()

        assert embed_dim % heads == 0, "Embedding dimension must be divisible by the number of heads"

        self.msa_block = AttentionBlock(embed_dim, heads, head_dropout)
        self.mlp_block = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=mlp_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_dim, out_features=embed_dim)
        )

    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x


class ViT(nn.Module):

    """
    Creates a Vision Transformer model.

    Parameters
    ----------
    image_size : int
        Size of the input image.
    patch_size : int
        Size of the patches.
    emb_dim : int
        Embedding dimension per token.
    depth : int
        Number of transformer blocks.
    heads : int
        Number of attention heads.
    mlp_dim : int
        Size of the feedforward layer in the transformer's MLP.
    dropout : float
        Dropout rate
    dropout_head : float
        Dropout rate for the attention heads.
    classes: list | tuple
        List of classes for the model to predict.
    custom_architecture: bool
        Whether to use the custom architecture or the original ViT architecture.


    Returns
    -------
    torch.nn.Module for a Vision Transformer model.
    """

    def __init__(
            self,
            image_size: int = 224,
            output_shape: int = 3,
            patch_size: int = 16,
            emb_dim: int = 768,
            depth: int = 12,
            heads: int = 12,
            mlp_dim: int = 3072,
            dropout: float = 0.1,
            dropout_head: float = 0.1,
            custom_architecture: bool = True
    ):
        super().__init__()

        n_patches = (image_size / patch_size) ** 2
        assert n_patches.is_integer(), \
            f'Image size must be divisible by patch size. Image size: {image_size}, patch size: {patch_size}'

        if custom_architecture:
            transformer = TransformerEncoder
            transformer_kw = {
                'embed_dim': emb_dim, 'heads': heads, 'mlp_dim': mlp_dim,
                'dropout': dropout, 'head_dropout': dropout_head
            }
        else:
            transformer = nn.TransformerEncoderLayer
            transformer_kw = {
                'd_model': emb_dim, 'nhead': heads, 'dim_feedforward': mlp_dim,
                'dropout': dropout, 'activation': 'gelu', 'batch_first': True, 'norm_first': True
            }

        self.embeddings_block = PatchEncoder(emb_dim, patch_size, int(n_patches), dropout=dropout)
        self.transformer_encoders = nn.Sequential(
            *[transformer(**transformer_kw) for _ in range(depth)]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=emb_dim),
            nn.Linear(in_features=emb_dim, out_features=output_shape)
        )

        self.summary = MethodType(utils.summary((1, 3, image_size, image_size)), self)

    def forward(self, x):
        x = self.embeddings_block(x)
        x = self.transformer_encoders(x)
        x = self.mlp_head(x[:, 0, :])  # Apply MLP on the CLS token only
        return x


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


def fine_tune_on_classes(pretrained_model: EfficientNet,
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
    pretrained_model.summary = fun  # type: ignore

    return pretrained_model.to(device)
