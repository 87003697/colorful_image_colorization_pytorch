# colorful image colorization pytorch

* This is a pytorch implementation of paper  [Colorful Image Colorization](https://arxiv.org/pdf/1603.08511.pdf), for [EECS 6691](https://sites.google.com/site/mobiledcc/advanceddeeplearning)  course presentation.
* Notice that in the [official repo](https://github.com/richzhang/colorization), only the demo code was uploaded. Other implementation repositories contain errors in loss function, preprocessing and postprocessing, so I rewrite the code using pytorch.

## Contribution
* To my knowledge, it's the only implementation including both training and inference using pytorch
* Model can be trained on both [ImageNet](http://www.image-net.org/) and other dataset, including [coco](https://cocodataset.org/#home).

## Usage
### Train
Link to your dataset (ImageNet or other) using
```
$ cd data
$ ln -s <your_dataset_root> ./
```
Specify your target dataset in **train.py** [line 129 and line 130](https://github.com/87003697/colorful_image_colorization_pytorch/blob/66699bbd717ae2c894c260f5cc6ab58e4afcaac2/train.py#L129).
You should be very careful about the dataset format. Use [defined module](https://github.com/87003697/colorful_image_colorization_pytorch/blob/66699bbd717ae2c894c260f5cc6ab58e4afcaac2/train.py#L142) as your `Dataset`, if your dataset is constructed like ⬇️
```
|-- root
    |-- image1.jpg
    |-- image2.jpg
    |-- ...
```
Otherwise specify [ImageFolder](https://github.com/87003697/colorful_image_colorization_pytorch/blob/66699bbd717ae2c894c260f5cc6ab58e4afcaac2/train.py#L138) if the format is like ⬇️ 
```
|-- root
    |-- folder1
        |-- image1.jpg
        |-- image2.jpg
    |-- folder2
        |-- image1.jpg
        |-- image2.jpg
    |-- ...
```
Then you can set off to training using
```
$ python train.py
```
PS: for other configuration of training, see [argument setup](https://github.com/87003697/colorful_image_colorization_pytorch/blob/66699bbd717ae2c894c260f5cc6ab58e4afcaac2/train.py#L102) 
### Performance
Due to time limit, at this moment the model is under training. According to [original paper](https://arxiv.org/pdf/1603.08511.pdf), the model should be trained for 500k+ iterations, which would spend several days or so. Feedback will be released once the training is finished. But now you can check the loss curves.
![train_loss_curve](https://github.com/87003697/colorful_image_colorization_pytorch/blob/main/images/train_loss_batches.png)![val_loss_curve](https://github.com/87003697/colorful_image_colorization_pytorch/blob/main/images/val_loss_epoches.png)
<figure class="half">
    <img src="https://github.com/87003697/colorful_image_colorization_pytorch/blob/main/images/train_loss_batches.png">
    <img src="https://github.com/87003697/colorful_image_colorization_pytorch/blob/main/images/val_loss_epoches.png">
</figure>

### Demo
Open **demo.ipynb**, choose either to inference with pre-saved model or your trained one.
Examples
![image](https://github.com/87003697/colorful_image_colorization_pytorch/blob/main/images/4611612935377_.pic_hd.jpg)

 


