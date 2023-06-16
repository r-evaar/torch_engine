import torch
import types
import requests
import tarfile
from tqdm import tqdm
from zipfile import ZipFile
from PIL import Image
from os import listdir
from os.path import basename
from pathlib import Path
from math import ceil
from joblib.externals.loky.backend.context import get_context
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose, ToTensor

"""
Contains functions for creating and managing PyTorch Datasets &  DataLoader
for image classification problems.
"""

 
def _write_to_file(f, response):
    block_size = 1024**2
    unit = 'MiB'
    total_size = round(int(response.headers.get("content-length", 0)) / block_size, 2)

    print(f'Total size = {total_size} {unit}')

    bar = tqdm(
        total=ceil(total_size), unit=unit,
        position=0, leave=True)

    for chunk in response.iter_content(chunk_size=block_size):
        f.write(chunk)
        bar.update()

    bar.close()
    assert bar.n == ceil(total_size), 'Download Failure: downloaded file is corrupted'


def download_data(directory: str,
                  url: str,
                  dataset_name: str = 'unnamed_dataset',
                  ):

    """
    Args:
        directory:
            Path to the directory where the data will be downloaded
        url:
            Link to a zip file containing the dataset.
        dataset_name:
            Name of the dataset folder. Contains the training, validation,
            and/or testing folders.
    """
    # url should include the filename of a compressed data file at the end.
    # Supported file formats are shown below.
    f_name = basename(url)
    supported = {
        '.gz':       (tarfile.open, 'r:gz'),
        '.xz':       (tarfile.open, 'r:xz'),
        '.tar':      (tarfile.open, 'r:'),
        '.zip':      (ZipFile, 'r')
    }
    f_frmt = Path(f_name).suffix
    try:
        handler, h_frmt = supported[f_frmt]
    except KeyError:
        raise Exception(f"Unsupported file format '{f_frmt}'.")

    # Get url response
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Verify directory
    data_path = Path(directory) / dataset_name
    if data_path.is_dir():
        print(f"[INFO] '{dataset_name}' dataset already exist. Download skipped.")
        return data_path

    print(f"[INFO] Did not find '{data_path}' directory. Creating one:")
    download_file = data_path / f'data_file{f_frmt}'

    # Download
    data_path.mkdir(parents=True, exist_ok=True)
    with open(download_file, 'wb') as f:
        print(f"[INFO] Downloading '{dataset_name}' dataset from '{url}':")
        _write_to_file(f, response)

    # Extract
    with handler(download_file, h_frmt) as f:
        print(f"[INFO] Extracting '{download_file}' .. ", end='')
        f.extractall(data_path)
        print('done.')

    # Delete
    download_file.unlink()
    return data_path


class ClassificationSet(Dataset):

    """
    Args:
        root:
            Path to the dataset
        transform:
            Transformation applied on X
        classes:
            List of dataset classes
        frmt:
            Format of targeted images under the root
        dtype:
            data type of input tensors
    """
    
    def __init__(self, root: str | Path,
                 transform: Compose = None,
                 classes: list = None,
                 frmt: str = 'jpg',
                 dtype: torch.dtype = torch.float):
        super().__init__()

        self.transform = transform
        self.dtype = dtype

        # verify class names if provided
        found_classes = listdir(root)
        if classes:
            assert found_classes, f'root directory {root} is empty.'
            assert sorted(found_classes) == sorted(classes), \
                f'Target mismatch: expected {classes}, found {found_classes}.'

        # Required for inference functions
        self.classes = found_classes
        self.class_to_idx = {name: torch.tensor(i, dtype=torch.long) for i, name in enumerate(found_classes)}

        # Perform a recursive file search under root for images with the given format
        if root is str: root = Path(root)
        self.image_filenames = list(Path(root).glob(f'*/*.{frmt}'))
        assert self.image_filenames, f"No image files with {frmt} format were found."

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, i):

        if i >= len(self):
            IndexError(f'{i} sample is out of dataset range.')

        # Get PIL image object at i
        filename = self.image_filenames[i]
        x = Image.open(fp=filename)
        if self.transform:
            x = self.transform(x)

        # Datasets & DataLoaders will always return tensors for both input & target
        if not torch.is_tensor(x): x = ToTensor()(x)
        x = x.type(self.dtype)
        y = self.class_to_idx[filename.parent.stem]

        return x, y


def _restart(self):
    # An added method to DataLoader objects to clear persistent workers if any

    # Shutdown persistent workers in active subprocesses
    if self._iterator and self.persistent_workers:
        if self._iterator._workers:
            print(f'Shutting down workers at {[worker.pid for worker in self._iterator._workers]}')
            self._iterator._shutdown_workers()

    # Collect the important attributes of the targeted DataLoader -> args
    args = {
        'shuffle': isinstance(self.sampler, torch.utils.data.sampler.RandomSampler),
    }
    attr = [
        'dataset', 'batch_size', 'pin_memory', 'pin_memory_device',
        'num_workers', 'persistent_workers', 'multiprocessing_context'
    ]
    for arg in attr:
        args[arg] = self.__getattribute__(arg)

    # Initialize a new DataLoader and assign it's state to this DataLoader
    # (because shutting down the workers disables it)
    self.__dict__.update(DataLoader(**args).__dict__)


def create_dataloaders(
        directories: list,
        transforms: None | list = None,
        splits: None | list = None,
        shuffles: list | bool = False,
        batch_size: int = 1,
        num_workers: int = 0,
        device: str = 'cpu'
):
    """Creates DataLoader objects

    Takes in a training directory and testing directory path and turns them into
    PyTorch Datasets and then into PyTorch DataLoaders

    Args:
        directories:
            A list of strings for the paths to the dataset subdirectories.
        transforms:
            A list of torchvision transforms to perform on the dataset subdirectories.
            Must be the same size as `directories`
        splits:
            A list of tuples where each specifies how each directory should be
            split. Affects the number of output DataLoaders.
        shuffles:
            A list of booleans for the shuffling option for each dataset
        batch_size:
            Number of samples per batch in each DataLoader.
        num_workers:
            Number of workers per DataLoader.
        device:
            Name of device where data tensors are allocated.

    Returns:
        A tuple of (*loaders, classes). Where classes is a list of the target classes.

    Example:
         train_loader, test_loader, class_names = create_dataloaders(
            train_dir=path/to/train_dir,
            test_dir=path/to/test_dir,
            batch_size=32,
            num_workers=4,
            device='cuda',
            splits=[(0.75, 0.25), None],
            shuffles=[True, False]
            )
        """

    # Verify that given kwargs (of list type) have the same lengths
    n_d = len(directories)
    if not transforms: transforms = [None] * n_d

    n_t = len(transforms)
    assert n_d == n_t, f"Expected {n_d} transforms for {n_d} subsets, Got {n_t}"

    n_s = len(splits)
    assert n_d == n_s, f"Expected {n_d} splits for {n_d} subsets, Got {n_s}"

    if shuffles is not list: shuffles = [shuffles] * n_d
    n_sh = len(shuffles)
    assert n_d == n_sh, f"Expected {n_d} shuffle configs for {n_d} subsets, Got {n_sh}"

    dataset = []  # A dataset is the parent of all subsets of which each will have a DataLoader
    shuffles_extended = []  # Extend the shuffles list in case of one or more random_split

    # Temporal variables to raise target mismatch error
    # (different # of classes in different subsets)
    classes = None
    prev_directory = ''

    for directory, transform, split, shuffle in zip(directories, transforms, splits, shuffles):
        # Append a subset and its shuffling setting
        subset = [ClassificationSet(root=directory, transform=transform)]
        shuffle = [shuffle]

        # Verify against previous subset
        classes_new = subset[0].classes
        if classes:
            assert classes_new == classes, \
                f"Target mismatch: found {classes} in '{prev_directory} and {classes_new} in '{directory}'."
        classes = classes_new
        prev_directory = directory

        # If a subset is to be split further, duplicate it's split settings (n = # of splits)
        if split:
            subset = random_split(*subset, split)
            shuffle = shuffle*len(subset)

        # Add the subset(s) and its(their) configuration
        dataset.extend(subset)
        shuffles_extended.extend(shuffle)

    # Extra settings required when using multiprocessing (num_workers > 0)
    com_kw = {
        'pin_memory': True,
        'pin_memory_device': device,
        'num_workers': num_workers,
        'persistent_workers': True,
        'multiprocessing_context': get_context('loky')
    } if num_workers > 0 else {}

    # Add one DataLoader for each subset
    loaders = []
    for subset, shuffle in zip(dataset, shuffles_extended):
        loader = DataLoader(dataset=subset, batch_size=batch_size, shuffle=shuffle, **com_kw)
        loader.restart = types.MethodType(_restart, loader)
        loaders.append(loader)

    for i, loader in enumerate(loaders):
        print(f'Loader#{i}: {len(loader)*batch_size} samples')
    print(f'Target: {classes}')

    return *loaders, classes
