# Code for several CIL Methods

## Dataset

1. For CIFAR-100, the dataset needs to be placed in the `./data/CIFAR/` . Otherwise, it will be automatically downloaded by PyTorch.

2. For TinyImageNet, the dataset needs to be placed in the `data/tiny-imagenet-200/` . The dataset can be find in https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet.

   Code to read TinyImageNet will be like:

   ```python
   torchvision.datasets.ImageFolder("./data/tiny-imagenet-200/train/")
   torchvision.datasets.ImageFolder("./data/tiny-imagenet-200/val/")
   ```

## Run

```bash
python main.py --config=exps/[CONFIG_FILE].json
```

### Common Hyperparameters

- `prefix`: `str` The prefix of logs and saved filename.
- `dataset`: `str`
  - `"cifar100"`
  - `"tinyimagenet200"`

- `shuffle`: `bool` Whether to shuffle the learning order of classes.
- `init_cls`: `int` The number of classes in the initial stage.
- `increment`: `int` The number of classes in each incremental stage.
- `model_name`: `str` The incremental learning method to run. See `models/` for detail.
- `convnet_type`: `str` Which backbone to use. See `convs/` and `utils/inc_net.py` for detail.
- `initial_model_path`: `str | None` If set, the initial stage will be skipped. Instead, the model will load weights from this path. It will be helpful when only parameters of incremental stages are changed.
- `device`: `list of str` Multi-GPU training is currently unavailable, so please only specify the GPU that you intend to use.
- `seed`: `list of int`  Train using each seed in the list sequentially. [1993] to reproduce our result.
- `init_epochs`: `int` Epochs for the initial stage.
- `init_lr`: `float` Learning rate for the initial stage.
- `init_weight_decay`: `float` Weight decay for the initial stage.
- `epochs`: `int` Epochs for the incremental stage.
- `lr`: `float` Learning rate for the incremental stage.
- `weight_decay`: `float` Weight decay for the incremental stage.
- `batch_size`: `int`
- `num_workers`: `int`
- `pin_memory`: `bool`
- `ncm_type`: `str` The way to calculate ncm distance.
  - `"euclidean"`: euclidean distance
  - `"cosine"`: cosine similarity
- `generator`: `str` How to generate the fake feature vectors of old classes.
  - `"oversampling"`
  - `"noise"`
  - `"translation"`: See [FeTrIL](https://github.com/GregoirePetit/FeTrIL).


**Other parameters can find in `models/[METHOD].py`**

## Our code references the the following repository:

[PyCIL](https://github.com/G-U-N/PyCIL), [FeCAM](https://github.com/dipamgoswami/FeCAM)

