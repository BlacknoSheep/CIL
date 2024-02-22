# Code for FRWF: A Class-Incremental Learning Method Based on Feature Reprojection and Weight Fusion

## Dataset

1. For CIFAR-100, the dataset needs to be placed in the `./data/CIFAR/` . Otherwise, it will be automatically downloaded by PyTorch.

2. For TinyImageNet, the dataset needs to be placed in the `data/tiny-imagenet-200/` . The dataset can be find in https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet.

   Code to read TinyImageNet will be like:

   ```python
   torchvision.datasetsImageFolder("./data/tiny-imagenet-200/train/")
   torchvision.datasetsImageFolder("./data/tiny-imagenet-200/val/")
   ```

## Run

```python
python main.py --config=exps/momentum.json
```

**Hyperparameters**

- `init_cls`: The number of classes in the initial stage.
- `increment`: The number of classes in each incremental stage.
- `initial_model_path`: If set, the initial stage will be skipped. Instead, the model will load weights from this path. It will be helpful when only parameters of incremental stages are changed.
- `reprojector`: Whether to use feature reprojection.
- `momentum`: Rate of Weight Fusion.
- `generator`: How to generate the fake feature vectors of old classes. Choose from `oversampling`, `translation`, and `noise`.

## Our code references the the following repository:

[PyCIL](https://github.com/G-U-N/PyCIL), [FeCAM](https://github.com/dipamgoswami/FeCAM)

