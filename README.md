# Few-Shot Learning with a Strong Teacher

This is the repository for "Few-Shot Learning with a Strong Teacher" [[ArXiv]](https://arxiv.org/abs/1812.03664) in PyTorch. If you use any content of this repo for your work, please cite the following bib entry:

    @article{ye2021few,
        author    = {Han-Jia Ye and
                    Lu Ming and
                    De-Chuan Zhan and
                    Wei-Lun Chao},
        title     = {Few-Shot Learning with a Strong Teacher},
        journal   = {CoRR},
        volume    = {abs/2107.00197},
        year      = {2021}
    }

The full code will be released soon.

## LastShot

Few-shot learning (FSL) aims to train a strong classifier using limited labeled examples. Many existing works take the meta-learning approach, sampling few-shot tasks in turn and optimizing the few-shot learner's performance on classifying the query examples. In this paper, we point out two potential weaknesses of this approach. First, the sampled query examples may not provide sufficient supervision for the few-shot learner. Second, the effectiveness of meta-learning diminishes sharply with increasing shots (i.e., the number of training examples per class). To resolve these issues, we propose a novel objective to directly train the few-shot learner to perform like a strong classifier. Concretely, we associate each sampled few-shot task with a strong classifier, which is learned with ample labeled examples. The strong classifier has a better generalization ability and we use it to supervise the few-shot learner. We present an efficient way to construct the strong classifier, making our proposed objective an easily plug-and-play term to existing meta-learning based FSL methods. We validate our approach in combinations with many representative meta-learning methods. On several benchmark datasets including miniImageNet and tiredImageNet, our approach leads to a notable improvement across a variety of tasks. More importantly, with our approach, meta-learning based FSL methods can consistently outperform non-meta-learning based ones, even in a many-shot setting, greatly strengthening their applicability.

<img src='lastshot.png' width='940' height='280'>

## Prerequisites

The following packages are required to run the scripts:

- [PyTorch-1.4 and torchvision](https://pytorch.org)

- Package [tensorboardX](https://github.com/lanpa/tensorboardX)

- Dataset: please download the dataset and put images into the folder data/[name of the dataset, miniimagenet or cub]/images

- Pre-trained weights: please download the [pre-trained weights](https://drive.google.com/open?id=14Jn1t9JxH-CxjfWy4JmVpCxkC9cDqqfE) of the encoder if needed. The pre-trained weights can be downloaded in a [zip file](https://drive.google.com/file/d/1XcUZMNTQ-79_2AkNG3E04zh6bDYnPAMY/view?usp=sharing).

## Dataset

### MiniImageNet Dataset

The MiniImageNet dataset is a subset of the ImageNet that includes a total number of 100 classes and 600 examples per class. We follow the [previous setup](https://github.com/twitter/meta-learning-lstm), and use 64 classes as SEEN categories, 16 and 20 as two sets of UNSEEN categories for model validation and evaluation, respectively.

Check [this](https://github.com/Sha-Lab/FEAT/blob/master/data/README.md) for details of data downloading and preprocessing.

## Code Structures
To reproduce our experiments with FEAT, please use **train_fsl.py**. There are four parts in the code.
 - `model`: It contains the main files of the code, including the few-shot learning trainer, the dataloader, the network architectures, and baseline and comparison models.
 - `data`: Images and splits for the data sets.
 - `saves`: The pre-trained weights of different networks.
 - `checkpoints`: To save the trained models.

## Acknowledgment
We thank the following repos providing helpful components/functions in our work.

- [FEAT](https://github.com/Sha-Lab/FEAT)

