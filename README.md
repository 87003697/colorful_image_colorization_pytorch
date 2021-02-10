# colorful image colorization pytorch

* This is a pytorch implementation of paper  [Colorful Image Colorization](https://arxiv.org/pdf/1603.08511.pdf).
* Notice that in the [official repo](https://github.com/richzhang/colorization), only the demo code was uploaded. Other implementation repositories contain errors in loss function, preprocessing and postprocessing, so I rewrite the code using pytorch.

## Contribution
* To my knowledge, it's the only implementation for both training and inference using pytorch
* Model is trained on both [ImageNet](http://www.image-net.org/) and other dataset, including [Coco](https://cocodataset.org/#home).

## Usage
### Train
```
$ python train.py
```
### Demo
open 

## TBD
