import torch
import os
import cpuinfo
import json
from .data_setup import create_dataloaders
from . import utils
from os import cpu_count
from torch import nn
from time import time as tic
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor, Compose, TrivialAugmentWide
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Precision, Recall, ROC, AUROC, ConfusionMatrix, F1Score, Accuracy
from mlxtend.plotting import plot_confusion_matrix
from random import sample
from numpy import ceil, expand_dims, ndarray, asarray
from datetime import datetime
from getpass import getuser
from types import MethodType


"""
Project Engine - contains: 
- Classes for managing and monitoring the train/eval process of nn Modules
- Functions for managing compute resources
- Functions for running automated experiments 
"""


def initialize() -> str:

    """
    Recognizes system support for CUDA devices. Prints and returns the available
    device.

    Returns:
        A str alias for the device where tensors are allocated.

    Use Example:
        device = initialize()
        print(device)
        'cuda'
    """

    if torch.cuda.is_available():
        device, device_name = 'cuda', torch.cuda.get_device_name()
    else:
        device, device_name = 'cpu', cpuinfo.get_cpu_info()['brand_raw']
    print(f'[CONNECTED] {device_name}')

    return device


def clear_processes():

    """
    Used when 'zombie processes' are created, typically due to destroyed DataLoader
    objects with persistent workers. This function terminates all subprocesses
    spawned by torch in the parent process.
    """

    active_pools = torch.multiprocessing.active_children()
    if active_pools:
        print('clearing active processes:', end=' ')
        for i, pool in enumerate(active_pools):
            print(f'[{i}]', end='')
            pool.terminate()
        print(' [DONE].')


class ClassTrainer:

    """
    A training manager for a classification model that initializes the
    training parameters, carries out the training, then shows and saves
    the results.

    Args:
        model: nn.Module classifier to train
        criterion: Loss function
        optim: Loss function optimizer
        lr: Learning rate
        lr_decay: Gamma value for an exponentially decaying learning rate
        writer: Write to tensorboard during training with a SummaryWriter
        verbose: Write to terminal during training
        progress: Save progress in a dictionary during training
    """

    def __init__(
            self, model: nn.Module,
            criterion=nn.CrossEntropyLoss(),
            optim=torch.optim.Adam,
            lr: float = 0.001,
            lr_decay: float = 0.975,
            writer=True,
            verbose=True,
            progress=True
    ):

        self.model = model
        self.device = list(model.parameters())[0].device

        self.optimizer = optim(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_decay)
        self.criterion = criterion
        self.writer = writer
        self.verbose = verbose

        d = lambda: {'x': [], 'loss': [], 'acc': []}
        self.progress = {'batch': d(), 'train': d(), 'val': d(), 'iter': 0} if progress else None
        self.writer_fn = SummaryWriter if writer else None

        self.first_batch = None
        self.m = None
        self.epoch = 0

    def verbose_step(self, update, *args):

        if not self.verbose:
            return

        if update == 'epoch_start':
            i, epochs = args
            print(f'[{i+1:0{len(str(epochs))}}]', end=' ')

        elif update == 'batch':
            i, n, n_b, loss = args  # iteration, progress checkpoint, total # of batches, loss value

            p = f' [{(i+1)*100/n_b:05.1f}%]'  # Epoch progress percentage | Example: [078.5%]
            v = f" batch_loss={loss:.2f}"  # Loss @ iteration i | Example: batch_loss=0.93
            b = '' if self.first_batch else '\b' * (len(v) + len(p))  # Backspace array.
                                                                      # None @ first iteration of current epoch
            self.first_batch = False  # Resets to True in the start of next epoch

            # Display batch progress
            print(f'{b}', end='')
            print('#' if i % n == 0 else '', end='')
            print(f'{p}{v if i < (n_b-1) else ""}', end='', flush=True)

        elif update == 'train':
            e_loss, e_acc, b_t, t0 = args  # Epoch loss, epoch accuracy, batch start time, epoch start time
            print(f' | train: l={e_loss:.2f}, a={e_acc:06.2f}% [{b_t:.2f}/{tic()-t0:.2f}s]', end=' ')

        elif update == 'val':
            loss, acc = args  # validation loss, validation accuracy
            print(f'| val: l={loss:.2f}, a={acc:06.2f}%', end=' ')

        elif update == 'lr':
            print(f'| lr={self.scheduler.get_last_lr()[0]:f}')  # Last learning rate

    def progress_step(self, tag, loss, acc):

        if not self.progress:
            return

        if tag == 'first':
            losses, accs = loss, acc  # inputs are arrays
            tags = ['train', 'val'] if len(losses) == 2 else ['train']
            for loss, acc, tag in zip(losses, accs, tags):
                self.progress_step(tag, loss, acc)
            return

        if tag == 'batch':
            self.progress['iter'] += 1
            loss = loss.item()

        self.progress[tag]['x'].append(self.progress['iter'])
        self.progress[tag]['loss'].append(loss)
        self.progress[tag]['acc'].append(acc)

    def writer_step(self, tag, loss, acc):

        if not self.writer:
            return

        if tag == 'first':
            losses, accs = loss, acc
            tags = ['train', 'val'] if len(losses) == 2 else ['train']
            for loss, acc, tag in zip(losses, accs, tags):
                self.writer_step(tag, loss, acc)
            return

        self.writer.add_scalar(tag=f"Loss/{tag}",
                               scalar_value=loss,
                               global_step=self.epoch)
        self.writer.add_scalar(tag=f"Accuracy/{tag}",
                               scalar_value=acc,
                               global_step=self.epoch)

    def first_eval(self, *loaders):
        print("[INFO] Initializing Training Process - First Evaluation")
        losses = []
        accs = []
        for loader in loaders:
            if not loader:
                continue  # In case no validation loader
            loss = 0
            acc = 0
            m = len(loader.dataset)
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                yp = self.model(x)

                r = x.shape[0]/m
                loss += self.criterion(yp, y).item() * r
                acc += self.accuracy(yp, y) * r
            losses.append(loss)
            accs.append(acc)
        return losses, accs

    def run(self, train_loader, val_loader=None, epochs=30,
            experiment_name=None, model_name="unnamed_model", extra=None):

        if not experiment_name:
            experiment_name = datetime.now().strftime("%H-%M-%S")

        run_title = f'{datetime.now().strftime("%Y-%m-%d")}_{getuser()}'
        # Tensorboard writer initialization
        if self.writer_fn:
            log_dir = os.path.join('runs', run_title, experiment_name, model_name)
            if extra: log_dir = os.path.join(log_dir, extra)

            print(f'[INFO] Initializing Tensorboard for:\n\t{log_dir}')

            self.writer = self.writer_fn(log_dir=log_dir)
            self.writer.add_graph(model=self.model, input_to_model=next(iter(train_loader))[0].to(self.device))

        # First run to evaluate network prior to training (and start persistent DataLoaders if applicable)
        losses, accs = self.first_eval(train_loader, val_loader)
        self.progress_step('first', losses, accs)
        self.writer_step('first', losses, accs)

        # Main Training Loop
        self.m = len(train_loader.dataset)
        print(f"[INFO] Training Started.\n{'-'*20}")
        t = tic()
        for i in range(epochs):
            self.epoch = i+1

            # Training & Validation Step
            self.verbose_step('epoch_start', i, epochs)
            self.run_epoch(train_loader)
            self.val_epoch(val_loader) if val_loader else None

            # Learning Rate Step
            self.verbose_step('lr')
            self.scheduler.step()

        print(f'{"-"*20}\n[INFO] Training Completed.\n\tTotal time = {tic()-t:.2f}s')

        # Tensorboard writer finalization
        if self.writer:
            self.writer.flush()

        # Restart loaders with persistent workers
        for loader in [train_loader, val_loader]:
            if loader.persistent_workers:
                loader.restart()

        return run_title

    def run_epoch(self, loader):
        self.model.train()
        e_loss = 0
        e_acc = 0

        # Verbose Requirements
        n_b = len(loader)
        n = ceil(n_b/10)
        self.first_batch = True

        e_time_0 = tic()
        b_avg_time = 0
        for i, (x, y) in enumerate(loader):
            t1 = tic()
            x, y = x.to(self.device), y.to(self.device)
            yp = self.model(x)

            loss = self.criterion(yp, y)
            acc = self.accuracy(yp, y)

            r = x.shape[0]/self.m
            e_loss += loss.item() * r
            e_acc += acc * r

            self.verbose_step('batch', i, n, n_b, loss)
            self.progress_step('batch', loss, acc)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            t2 = tic()-t1
            b_avg_time = t2 * i/n_b

        self.verbose_step('train', e_loss, e_acc, b_avg_time, e_time_0)
        self.progress_step('train', e_loss, e_acc)
        self.writer_step('train', e_loss, e_acc)


    def val_epoch(self, loader):
        self.model.eval()
        loss, acc, m = 0, 0, len(loader.dataset)
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            yp = self.model(x)

            r = x.shape[0]/m
            loss += self.criterion(yp, y).item()*r
            acc += self.accuracy(yp, y)*r

        self.verbose_step('val', loss, acc)
        self.progress_step('val', loss, acc)
        self.writer_step('val', loss, acc)

    @staticmethod
    def accuracy(yp, y):
        tp = torch.eq(
            torch.argmax(nn.functional.softmax(yp, dim=1), dim=1),
            y
        ).sum().item()
        return tp/len(y) * 100

    def evaluate(self, loader, classes):

        loss = 0
        m = len(loader.dataset)
        n = len(loader)

        per = f'[00.0%]'
        print(per, end='', flush=True)

        y_all = []
        yp_all = []

        self.model.eval()
        with torch.inference_mode():
            for i, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)

                r = x.shape[0]/m

                yp = self.model(x)
                loss += self.criterion(yp, y) * r

                y_all.append(y)
                yp_all.append(yp)

                print('\b'*len(per), end='')
                per = f'[{(i+1)*100/n:05.1f}%]'
                print(per, end='', flush=True)
        print('\n')
        loader.restart()

        yp_all = torch.cat(yp_all, dim=0)
        y_all = torch.cat(y_all, dim=0)

        metrics = [
            metric(task='multiclass', average='none', num_classes=len(classes)).to(self.device)
            for metric in [Accuracy, Precision, Recall, F1Score, AUROC, ROC, ConfusionMatrix]
        ]
        metrics[0].average = 'micro'  # For Accuracy scores

        acc, precision, recall, f1, auc, roc, confusion_matrix = [metric(yp_all, y_all) for metric in metrics]
        means = [m.sum()/len(m) for m in (precision, recall, f1, auc)]
        tags = ['Precision\t', 'Recall\t\t', 'F1 Score\t', 'AUC\t\t\t']

        print('Mean Average Values:')
        print('-'*20)
        print(f'Loss\t\t{loss:.4f}')
        print(f'Acc\t\t\t{acc:.4f}')
        for tag, mean in zip(tags, means):
            print(f'{tag}{mean:.4f}')

        matrix_fig, _ = plot_confusion_matrix(
            conf_mat=confusion_matrix.detach().cpu().numpy(),
            class_names=classes,
            cmap='BuGn'
        )

        if isinstance(self.writer, SummaryWriter):
            matrix_fig.savefig(Path(self.writer.get_logdir())/'confusion_matrix.png')
            self.writer.add_figure('Confustion Matrix Fig', matrix_fig)
            self.writer.add_pr_curve('Precision Recall',
                                     y_all, torch.argmax(nn.functional.softmax(yp_all, dim=1), dim=1))

            for i, target in enumerate(classes):
                text = f"Precision: {precision[i]:.3f}"
                text += f" | Recall: {recall[i]:.3f}"
                text += f" | F1 Score: {f1[i]:.3f}"
                text += f" | AUC: {auc[i]:.3f}"
                self.writer.add_text(target, text)

            self.writer.flush()




def inference(model: nn.Module,
              target: str,
              classes: list,
              transform=None,
              device='cpu',
              n=10):
    target = Path(target)
    supported = ['jpg', 'jpeg', 'png']

    filenames = []
    if target.is_dir():
        for ext in supported:
            filenames.extend(list(target.glob(f'**/*.{ext}')))
        assert filenames, f"Could not find any images in '{filenames}'"
    elif target.is_file():
        ext = target.suffix.split('.')[1]
        assert ext in supported, f"Files with '{ext}' extension are not supported"
        filenames.append(target.__str__())
        n = 1
    else:
        raise FileNotFoundError(f"cannot find '{target}'")

    if len(filenames) > n:
        filenames = sample(filenames, k=n)

    n_c = min(5, n)
    n_r = int(ceil(n/n_c))
    fig, axes = plt.subplots(nrows=n_r, ncols=n_c, figsize=(20, 4*n_r))

    if not isinstance(axes, ndarray):
        axes = asarray([[axes]])
    if axes.ndim < 2:
        axes = expand_dims(axes, 0)

    model.eval()
    r = 0
    c = 0
    with torch.inference_mode():
        for i, image_file in enumerate(filenames):
            if c == n_c:
                c = 0
                r += 1

            image = Image.open(image_file)
            x = image
            if transform:
                x = transform(x)
            if not torch.is_tensor(x):
                x = ToTensor()(x)
            x = x.unsqueeze(0).to(device)
            yp = nn.functional.softmax(model(x), dim=1)
            idx = torch.argmax(yp, dim=1).item()
            label, prop = classes[idx], yp.squeeze()[idx]

            axes[r, c].imshow(image)
            axes[r, c].set_title(f'{label} ({prop:.2f})')
            axes[r, c].axis('off')

            c += 1

        if c < n_c:
            for c in range(c, n_c):
                axes[r, c].axis('off')


def start_experiments(experiments):

    device = initialize()

    stamp = datetime.now().strftime('%H_%M_%S')
    experiments_name = f'automated_experiment_{stamp}'

    n = len(experiments)
    fmt = f'0{len(str(n))}d'
    for i, experiment in enumerate(experiments):
        print('\n\n')
        exp_name = f'Exp_{i:{fmt}}'

        # Extract experiment configurations
        model_fn, weights = experiment['model'].extract()
        data_path = experiment['data'].extract()
        epochs = experiment['epochs']
        dropout = experiment['dropout']

        print(f'[INFO] Configurating {exp_name}')

        # Create & Freeze model for transfer learning
        model = model_fn(weights=weights)
        for param in model.features.parameters(): param.requires_grad = False

        # Transforms are created based on the pre-trained weights of the model
        auto_transform = weights.transforms()
        transforms = [
            Compose([TrivialAugmentWide(), auto_transform]),
            auto_transform
        ]

        # Expecting train/test sub_directory per the dataset in each experiment
        directories = [data_path/'train', data_path/'test']

        # Subdividing the train subset into train & val, resulting in a total of 3 loaders
        train_loader, val_loader, test_loader, classes = create_dataloaders(
            directories=directories,
            transforms=transforms,
            batch_size=32,
            num_workers=min(3, cpu_count()//3),
            device=device,
            splits=[(0.8, 0.2), None],
            shuffles=[True, False]
        )

        # Altering the model's head based on the dataset's # of classes and the experiment's dropout rate
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features=model.features[-1][-3].out_channels, out_features=len(classes))
        )

        # After model is finalized, move it to device
        model.to(device)

        # Initialize a ClassTrainer object for managing and monitoring the training progress of a classification model
        resize = [auto_transform.resize_size[0]]*2
        model.summary = MethodType(utils.summary(input_size=[1, 3, *resize]), model)
        trainer = ClassTrainer(model, progress=False)

        # Starting a single ClassTrainer run with the number of epochs specified for the experiment
        print(f"[INFO] Starting {exp_name}")
        run_title = trainer.run(
            train_loader, val_loader, epochs=epochs,
            experiment_name=experiments_name,
            model_name=exp_name
        )

        # Save configuration as JSON file & model as PT file to the same
        # path of the experiment's Tensorboard results
        print(f"[INFO] Saving {exp_name} configs and weights")
        exp_path = Path('runs') / run_title / experiments_name / exp_name

        config_file = exp_path / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(
                obj={k: v.__str__() for k, v in experiment.items()},
                fp=f, indent=2
            )

        model_file = exp_path / f"{exp_name}.pt"
        torch.save(model.state_dict(), f=model_file)
