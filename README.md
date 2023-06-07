# Torch Engine
modular engine to provide high-level interface for different CNN tasks

### [NOTE] Work in progress
Currently supported tasks:
* CNN multi-classification
___

A development prototype is available at `interface.ipynb` notebook

#### Supported Features:

1. Device-agnostic setup

```Python
device = engine.initialize()
```

2. Downloaded and extract multclass datasets

```Python
data_path = data_setup.download_data(directory=<project_dir/all_data_directory>, url=<url/to/file.zip*>, dataset_name=<dataset_name>)
```
* Supports `zip`, `tar`, `gz`, and `xz` formats.

3. Multiprocessing-Ready DataLoaders

```Python
loader_1_1, loader_1_2, loader_2, ..., classes = data_setup.create_dataloaders(
    directories=[dir_1, dir_2, ...], 
    transforms=[transf_1, transf_2, ...], 
    splits=[(ratio_1, ratio_2), None, ...], 
    shuffles=[True, False, ...],
    batch_size=<batch_size>, 
    num_workers=<worker_per_loader>, 
    device=<cuda|cpu>
)
```

4. Custom & modified pre-trained models

```Python
model = models.fine_tune_on_classes(<pre_trained_model>, target=<class_names_list>, input_size=(height, width), **kw)

model = TinyVGG(input_shape=(channels, height, width), classes=<pre_trained_model>, **kw)
```

5. Classification Training Manager

```Python
trainer = engine.ClassTrainer(model, **kw)
trainer.run(train_loader, val_loader, epochs=50, model_name='EffNetB1_50e', **kw)
```

6. Model Evaluation with Torchmetrics

```Python
trainer.evaluate(test_loader, <class_names_list>)
```

7. Inference and plotting of N samples

```Python
engine.inference(model, <path/to/image(s)>, <class_names_list>, n=4, **kw)
```
8. Automatic experiments manager

```Python
experiment_space = {
    'model':        models,
    'data':         [dataset_1, dataset_2],
    'epochs':       [5, 10],
    'dropout':      [0.1, 0.2, 0.3],
}

experiments = utils.prepare_experiments(experiment_space)
engine.start_experiments(experiments)
```
