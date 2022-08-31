# PushPull Convolutions
Push Pull Conv2D - Improving robustness of traditional convolutional neural networks (ConvNets) against typical non-adversarial attacks.


### Dataset 
We use 2 data sets - CIFAR-10 and ImageNet-100. 
The training of the ConvNets is performed on the clean images. 
Evaluation is done on images with several types and severities of common image corruptions.

| Data set characteristics                       | CIFAR-10 |   ImageNet-100   |
|------------------------------------------------|:--------:|:----------------:|
| # classes                                      |    10    |       100        |
| # training images per class                    |  5,000   | 1,285 (on avg.)  |
| # test images per class (clean)                |  1,000   |        50        |
| # test images per class (corrupted with noise) |  75,000  |      3,750       |

The corrupted versions of each test image is created according to the details mentioned [here](https://github.com/hendrycks/robustness). 
Overall there are 15 types of image corruptions and 5 levels of severity for each type.
This results in 75 corrupted versions of each test image.

### Experiments
The robustness of the PushPull Convolutions is evaluated on two different problems - image classification and image hashing. 
The backbone network used for the both the problems is ResNet50.


### Results


### Citation
If you find this useful in your research, please consider citing:
