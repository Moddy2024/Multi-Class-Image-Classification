# Multi-Class-Image-Classification
Trained a CNN to perform Multi-Class Image Classification. The CNN contains 5 layers consisting of Batch-Normalization on the first 4 layers and max pooling layers followed by a dropout layer of 30% then the Fully connected layer. It's been trained on CIFAR-10 dataset for 15 epochs on a GPU. It achieved a testing accuracy of 82% on a testing dataset of 10,000 images.
# Software requirements
* [PyTorch](https://pytorch.org/)
* [Matplotlib](https://matplotlib.org/)
* [PIL](https://pypi.org/project/Pillow/)
* [fast.ai](https://www.fast.ai/)
* [Torchvision](https://pytorch.org/vision/stable/index.html)
* [torchinfo](https://github.com/TylerYep/torchinfo)
# Key Files
* [training.ipynb](https://github.com/Moddy2024/Multi-Class-Image-Classification/blob/main/training.ipynb) - In this file you can see how the data has been downloaded, checking of the images, defining and training of the CNN and testing of the model.
* [test_data](https://github.com/Moddy2024/Multi-Class-Image-Classification/tree/main/test_data) - This folder contains some images that I have downloaded from the internet to check the prediction of model of single images.
* [trained_model](https://github.com/Moddy2024/Multi-Class-Image-Classification/tree/main/trained_model) - This folder contains the trained model that I have saved after training. It can also be used for transfer learning.
* [prediction.ipynb](https://github.com/Moddy2024/Multi-Class-Image-Classification/blob/main/prediction.ipynb) -  Using the trained model pth file from the above folder and images, this predicts the class of the image.
# Dataset
The dataset can be downloaded from [here](https://www.cs.toronto.edu/~kriz/cifar.html) or by running the 2nd cell of this notebook [training.ipynb](https://github.com/Moddy2024/Multi-Class-Image-Classification/blob/main/training.ipynb) it downloads directly from fast.ai. The CIFAR-10 dataset consists of 60000 (32x32) colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain exactly 5000 images from each class. The 10 classes are airplane, automobile(includes sedans, SUVs, things of that sort), bird, cat, deer, dog, frog, horse, ship, and truck(Big trucks only).

