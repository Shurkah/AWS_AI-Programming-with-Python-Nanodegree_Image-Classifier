import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sb
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image


# Function that loads a checkpoint and rebuilds the model

def load_checkpoint(path):
    checkpoint = torch.load(path)

    arch = checkpoint['arch']

    if arch == "vgg13":
        model = models.vgg13(pretrained=True)

    else:
        model = models.densenet121(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    optimizer = checkpoint['optimizer']
    inputs = checkpoint['inputs']
    outputs = checkpoint['outputs']
    hidden_units = checkpoint['hidden_units']
    epochs = checkpoint['epochs']

    return model


# Specifications: There is a function that successfully loads a checkpoint and rebuilds the model


# Function that pre-processes a PIL image for use in a PyTorch model

def process_image(path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    img = Image.open(path)
    img.resize((256, 256))

    # crop 224 x 224, IN THE CENTER OF THE IMAGE
    a = int((img.size[0] - 224) / 2)
    b = int((img.size[1] - 224) / 2)
    c = a + 224
    d = b + 224
    img = img.crop((a, b, c, d))

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    img = (((np.array(img) / 255) - means) / stds)

    img = img.transpose((2, 0, 1))

    return img


# Specifications: The process_image function successfully converts a PIL image into an object that can be used as input to a trained model


# Function that displays the image for checking

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


# Function that predicts the class from an image file

def predict(img, model, top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    img = torch.from_numpy(img).type(torch.cuda.FloatTensor)
    img = img.unsqueeze(dim=0)

    with torch.no_grad():
        model = model.to('cuda')
        output = model(img)
    probs = torch.exp(output)  # converting into a probability

    classes = probs.topk(top_k)
    return classes

# Specifications: The predict function successfully takes the path to an image and a checkpoint, then returns the top K most probable classes for that image
