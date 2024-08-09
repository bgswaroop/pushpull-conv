## PushPull Convolutions
#### Accepted at ICPR 2024
#### Quick Links
- [ICPR Supplementary Material](https://github.com/bgswaroop/pushpull-conv/blob/main/resources/ICPR_Supplementary_Material.pdf)
- [pre-print on arXiv](https://arxiv.org/abs/2408.04077)

### Paper Summary
PushPull-Conv: Improving the robustness of traditional convolutional neural networks (ConvNets) against typical image corruptions, by a simple architectural change. 
Replace the first Conv layer with the [PushPull-Conv unit](project/models/utils/push_pull_unit.py) to achieve robustness. The goal is to achieve robustness to unseen distributions during model learning.

#### Dataset 
Datasets used: CIFAR-10, ImageNet-100, ImageNet-200, and ImageNet-1k. 
Models were trained on the clean images, and tested on corrupted images (CIFAR10-C and ImageNet-C; [hendrycks et. al](https://github.com/hendrycks/robustness)). 
An example of the 15 types of corruption with 5 levels of severity is shown below:
![imagenet_c](resources/figure_imagenet_c.png)

#### Experiments & Results
Results on ImageNet are presented below. 
For other datasets and models please refer to the [paper](https://arxiv.org/abs/2408.04077).
In the Table below, **E** is the top1 clean error, **mCE** is mean corruption error and **R_net** is net reduction in error rate
when compared to the baseline. **R_net** is expressed as a percentage, and accounts for both **E** and **mCE** (Refer to Sec. 5.2 in the [paper](https://arxiv.org/abs/2408.04077) for further details).

| Variants of baseline models    |  E ↓  |  mCE↓ | R_net↑ |
|:-------------------------------|:-----:|------:|-------:|
| ResNet50 (baseline)            | **0.269** | 0.667 |   0.00 |
| ResNet50 + PushPull avg3 (PP3) | 0.282 | 0.645 |   1.67 | 
| ResNet50 + PushPull avg5 (PP5) | 0.276 | 0.645 |   2.79 |
| ResNet50 + AutoAug             | 0.269 | 0.630 |   6.86 |
| ResNet50 + AutoAug + PP3       | 0.283 | 0.604 |   9.05 |
| ResNet50 + AugMix              | 0.269 | 0.612 |  10.14 |
| ResNet50 + AugMix + PP3        | 0.287 | 0.590 |  10.97 |
| ResNet50 + PRIME               | 0.298 | 0.522 |  21.57 |
| ResNet50 + PRIME + PP3         | 0.306 | **0.500** |  **24.31** |

### Instructions to run the code
- The current implementation is suited to run on HPC Clusters, an example script is located [here](project/misc/jobs/run_jobs_habrok.sh).
- A simplified script is being developed to quickly run both the training and the predict flows.

### Citation
If you find this useful in your work, please consider citing:
<pre>
@misc{bennabhaktula2024push, 
    title={{PushPull-Net: Inhibition-driven ResNet} robust to image corruptions},   
    author={Guru Swaroop Bennabhaktula and Enrique Alegre and Nicola Strisciuglio and George Azzopardi},  
    year={2024},  
    eprint={2408.04077},  
    archivePrefix={arXiv},  
    primaryClass={cs.CV},  
    url={https://arxiv.org/abs/2408.04077},   
}
</pre>
