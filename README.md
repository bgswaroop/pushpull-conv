# PushPull Convolutions
PushPull-Conv: Improving the robustness of traditional convolutional neural networks (ConvNets) against typical image corruptions, by a simple architectural change. Replace the first Conv layer with the PushPull-Conv unit to achieve robustness.  

### Dataset 
We test our method on the following datasets: CIFAR-10, ImageNet-100, ImageNet-200, and ImageNet-1k. 
The training of the ConvNets is performed on the clean images. 
Evaluation is done on images with several types and severities of common image corruptions (CIFAR10-C and ImageNet-C; [hendrycks et. al](https://github.com/hendrycks/robustness)). In summary, there are 15 types of image corruption with 5 levels of severity for each kind. An example is shown below:

#todo: add an image showing image corruption

### Experiments
The robustness of the PushPull Convolutions is evaluated on two different problems - image classification
The backbone network used for the both the problems is ResNet50.

### Results


### Citation
If you find this useful in your research, please consider citing:
