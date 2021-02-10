# colorful image colorization pytorch

* This is a pytorch implementation of paper  [Colorful Image Colorization](https://arxiv.org/pdf/1603.08511.pdf), for [EECS 6691](https://sites.google.com/site/mobiledcc/advanceddeeplearning)  course presentation.
* Notice that in the [official repo](https://github.com/richzhang/colorization), only the demo code was uploaded. Other implementation repositories contain errors in loss function, preprocessing and postprocessing, so I rewrite the code using pytorch.

## Contribution
* To my knowledge, it's the only implementation for both training and inference using pytorch
* Model is trained on both [ImageNet](http://www.image-net.org/) and other dataset, including [Coco](https://cocodataset.org/#home).

## Usage
### Train
Link to your dataset (imagenet or other) using
```
$ cd data
$ ln -s <your_dataset_root> ./
```
Specify your target dataset in **train.py**.
```
$ python train.py
```
### Demo
Open **demo.ipynb**, choose either to inference with pre-saved model or your trained one. 

 

## TBD
