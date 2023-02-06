import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import json
import numpy as np
import functions
import argparse


parser = argparse.ArgumentParser(
    description='Argument parser for predicting'
)

parser.add_argument('input', default='./flowers/test/19/image_06170.jpg', action="store", type = str)
parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str)
parser.add_argument('--gpu', action="store", type=bool, default=True)
parser.add_argument('--top_k', action="store", type=int, default=5)
parser.add_argument('--category_names', action="store", default='cat_to_name.json')


args = parser.parse_args()
checkpoint = args.checkpoint
gpu = args.gpu
impath = args.input
top_k = args.top_k
category_names = args.category_names


# Use GPU if it's available
device = torch.device("cuda" if (torch.cuda.is_available() and gpu == True) else "cpu")

model = functions.load_checkpoint(checkpoint)


img = functions.process_image(impath)

probs, classes = functions.predict(img, model, top_k)
probs = np.array(probs[0])
print('Probabilities: ', ["%.2f" % p for p in probs])

classes = np.array(classes[0])

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
class_names = [cat_to_name[str(x)] for x in classes]
print('Class names: ', class_names)