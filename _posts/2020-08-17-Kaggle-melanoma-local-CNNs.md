---
title: "Melanoma Prediction Kaggle Contest - Part 3: Image Data"
date: 2020-08-12
tags: [kaggle, competition, CNN, neural networks]
mathjax: true
classes: wide
---

If you would prefer to go straight to the code to learn about my approach to the competition please [go here](https://github.com/sjhatfield/kaggle-melanoma-2020). The notebooks for the models on this page can be found [here](https://github.com/sjhatfield/kaggle-melanoma-2020/blob/master/notebooks/).

Previously: [Part 1](https://sjhatfield.github.io/2020-08-12-Kaggle-melanoma-contest-exploration) and [Part 2](https://sjhatfield.github.io/Kaggle-melanoma-tabular-model/).

Credit: the user [Roman](https://www.kaggle.com/nroman) has provided many public notebooks on Kaggle which were incredibly useful in developing my own code.

## Introduction

It is now time to train models which will use the patient image data. As we are using image data we will be using convolutional neural networks. I begin training some smaller models which can be done on my local machine and then test out training a larger model before sending the computation to the cloud so that larger image sizes can be used.

## Image Augmentations

It is a almost always a good idea to perform augmentations on the images before passing them to the model for training. This diversifies and increases the size of the training data and should help the model to generalize to more varied images. Augmentations should be used that are appropriate to the training set. If you are trying to classify images of dogs into different breeds then it would not make sense to reflect an image across the horizontal because you will not need to classify pictures of upside down dogs.

The code for my image augmentations can be found [here](https://github.com/sjhatfield/kaggle-melanoma-2020/blob/master/src/data/prepare_data.py#L12). The augmentations used were:

* Horizontal flip
* Vertical flip
* Shear in X
* Shear in Y
* Translate in X
* Translate in Y
* Rotation
* Apply auto-contrast
* Apply equlization (equalizes the histogram of the image to create a uniform distribution of greyscale pixels)
* Solarize (inverts all pixels above a certain threshold)
* Posterize (reduces the number of bits for each pixel)
* Change contrast
* Color balance (similar to settings on a television)
* Brightness
* Sharpness
* Cutout (cuts out a rectangular shape and fills with a fillcolor)

The images to be classified are photos of skin lesions taken directly from above. So most augmentations make sense as there is no fixed perspective.

The next thing to decide is how many of these to be applied and what parameters to be used with them. For example, by how many degrees do we rotate. I followed the advice from the paper [UniformAugment: A Search-free ProbabilisticData Augmentation Approach](https://arxiv.org/abs/2003.14348v1). The key takeaway from this paper is that, "a uniform sampling over the continuous space of augmentation transformations is sufficient to train highly effective models". With this is mind, to get an image from the training data to be passed to the network, a random sample of two augmentations was chosen from all that are possible above. Then each was performed uniformly at random with a magnitude randomly chosen from their range of possible magnitudes. Here is the code that explains how this happens:

```python
def __call__(self, img: PIL.Image) -> PIL.Image:
        """
        Select a random sample of augmentations where 
        self.operations determines how many and perform them
        on the image
        """
        operations = random.sample(list(self.augs.items()), self.operations)
        for operation in operations:
            augmentation, range = operation
            # Uniformly select value from range of augmentations
            magnitude = random.uniform(range[0], range[1])
            # Perform augmentation uniformly at random
            probability = random.random()
            if random.random() < probability:
                img = self.func[augmentation](img, magnitude)
        return img
```

To show this in action here is a sample of 20 images with their augmentations. Remember some of these will have no augmentation at all. These are the not the exact form of the images that are passed to the model as they have not been normalized according to the ImageNet database that most pre-trained neural networks are trained on. If I was to show normalized images they would be bright colors.

<center><img src="{{ site.url }}{{ site.baseurl }}/images/kaggle-melanoma/augmentation_example.png" class="center" alt="20 example augmented images to be passed to the model."></center>

## Software

I will be using PyTorch for the neural network implementation. The Python file where I create the networks and key functions required for training can be found [here](https://github.com/sjhatfield/kaggle-melanoma-2020/blob/master/src/models/model.py).

## Model Parameters

Despite using a relatively standard training, validation, testing strategy with fairly standard architectures there are still a number of paramters to be chosen for fitting. My choices with justification come next.

### Model Architecture

The architectures that I used on my local machine were Alexnet and Resnet18. These are fairly small convolutional neural networks, in the sense of the number of parameters to be trained. Once I moved the computation to the cloud, I used EfficientNets. I will briefly explain the idea behind them from the paper [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946). The authors, Mingxing Tan and Quoc V. Le, performed an architecture search where they scaled the three main measurements that determine the size of a network using the same scaling constant. The three dimensions of convolutional neural networks they scaled were:

* Width: the width or number of channels of each layer in the network
* Depth: the number of layers in the convolutional neural network
* Resolution: this effectively the size of the images fed into the model

By first finding an efficient baseline network EfficientNet-B0 they then scaled it larger and larger to achieve greater performance on ImageNet. Their basline network EfficientNet-B0 has around one fifth the number of parameters as ResNet-50 but higher accuracy. As a result of their paper, they found EfficientNet acritectures from B0 to B7 increasing in size and accuracy. For example, EfficientNet-B7 has around one eighth the number of parameters as GPipe, the previously best performing CNN on ImageNet.

So EfficientNets are not only more efficient than other networks commonly used, they are often more accurate. This made them a common choice in the competition amongst competitiors.

### Batch Size

This is the number of images sent to the network together for training. Generally, the higher the batch size the better so I sent as many could be handled by the GPU or CPU at once. For the larger networks I trained this had to be eigher 16 or 32 iamges.

### Loss Function

As this is a binary classification problem I used Cross Entropy loss which has the following formula

$H = - \frac{1}{N} \sum_{i=1}^N \left[ y_n \log \hat{y}_n + (1 - y_n ) \log (1 - \hat{y}_n) \right]$ 

where $$y_n$$ is the true label of y (0: benign, 1: malignant), $$\hat{y}_n$$ the predicted probability of $$y$$ being malignant and $$N$$ the number of total images in the loss calculation. This loss function punishes high $$\hat{y}$$ values when the actual label is $$0$$ and vice-versa.

### Optimization

I use the [Adam algorithm](https://arxiv.org/abs/1412.6980) for stochastic optimization. It is both computationally efficient and has little memory requirements. This is the most common optimization algorithm used in neural network training.

### Learning Rate

An initial learning rate of `1e3` $$= 0.001$$ is used but it is scheduled to decrease once the validation accuracy plateaus. This is implemented in PyTorch using

```python
optimizer = optim.Adam(net.parameters(), lr=1e3)
scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=2, factor=0.2)
```

where `mode='max'` means the learning rate will be decreased once the validation accuracy stops increasing, `patience=2` means we will allow only 2 epochs of non-increasing performance and  `factor=0.2` means the current learning rate will be multiplied by 0.2 for the decrease. Another learning rate schedular which performs well is [cyclic learning rates](https://arxiv.org/abs/1506.01186) which is also available in PyTorch.

### Sampling

I use PyTorchs `WeightedRandomSampler` to ensure the minority class (malignant) is sampled often enough to make sure it is $$50\%$$ of the training data. It is possible that it would be better to only sample until it makes up maybe $$20\%$$ of the data but I did not have enough time (and compute) to try this. I also wanted to use SMOTE with the image data but could not find a PyTorch implementation.

### Number of Epochs

For the local smaller networks I trained for 10 epochs and for the cloud ones 20. This was purely due to time and computation limits. I would have trained for longer if I had access to better hardware.

### Frozen Epochs

I left all layers of the models except the final frozen for the first 3 epochs. This is because I was using pre-trained networks which has their final network changed to match the cancer prediciton binary classification problem. The networks are pre-trained on ImageNet which contains images of all manner of different objects/animals/things. By freezing all but the last layer, we can relatively quickly get good performance on the cancer image data and them unfreeze the full network and train the lower layers to the specific task. This is called transfer learning.

### Early Stopping

This technique is to combat overfitting by stopping training if an increase in performance on the validation set is not found for a certain number of epochs in a row. I had this set to 3 for the local training and 5 for cloud training. This way a reduction in the learning rate can be tested and then training halted if better performance is not experienced with the lower learning rate.