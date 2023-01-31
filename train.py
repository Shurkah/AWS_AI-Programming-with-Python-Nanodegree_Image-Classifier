import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import functions
import argparse
import os

# Specifications: All the necessary packages and modules are imported in the first cell of the notebook


parser = argparse.ArgumentParser(
    description='Argument parser for training'
)
parser.add_argument('data_dir', action="store", default="./flowers")
parser.add_argument('--save_dir', action="store", default="./checkpoint.pth")
parser.add_argument('--arch', action="store", default="densenet121")
parser.add_argument('--learning_rate', action="store", type=float, default=0.003)
parser.add_argument('--epochs', action="store", type=int, default=1)
parser.add_argument('--dropout', action="store", type=float, default=0.2)
parser.add_argument('--hidden_units', action="store", type=int, default=512)
parser.add_argument('--gpu', action="store", type=bool, default=True)

args = parser.parse_args()
data_dir = args.data_dir
checkpath = args.save_dir
input_lr = args.learning_rate
arch = args.arch
gpu = args.gpu
input_epochs = args.epochs
dropout = args.dropout
hidden_units = args.hidden_units

# checkpoint_save_dir = args.save_dir
# if not os.path.exists(checkpoint_save_dir):
#     os.makedirs(checkpoint_save_dir)

# Loading data
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
transforms_train = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
# Specifications: torchvision transforms are used to augment the training data with random scaling, rotations, mirroring, and/or cropping


transforms_valid_test = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])

# Specifications: The training, validation, and testing data is appropriately cropped and normalized


# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=transforms_train)
test_data = datasets.ImageFolder(test_dir, transform=transforms_valid_test)
validation_data = datasets.ImageFolder(valid_dir, transform=transforms_valid_test)

# Specifications: The data for each set (train, validation, test) is loaded with torchvision's ImageFolder


# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
validloader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=True)

# Specifications: The data for each set is loaded with torchvision's DataLoader



# Use GPU if it's available
device = torch.device("cuda" if (torch.cuda.is_available() and gpu == True) else "cpu")


# Load a pre-trained network
if arch == "vgg13":
    model = models.vgg13(pretrained=True)
    inputs = 25088

else:
    model = models.densenet121(pretrained=True)
    inputs = 1024


    # Freeze the network's parameters
for param in model.parameters():
    param.requires_grad = False

# Specifications: A pretrained network such as VGG16 is loaded from torchvision.models and the parameters are frozen
# I chose densenet because it's not so big


# Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout

outputs = 102
model.classifier = nn.Sequential(nn.Linear(inputs, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_units, outputs),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=input_lr)

# For In [5]
model.to(device);

# Specifications: A new feedforward network is defined for use as a classifier using the features as input


# Train the classifier layers using backpropagation using the pre-trained network to get the features

epochs = input_epochs
steps = 0
running_loss = 0
print_every = 5

for epoch in range(epochs):
    for images, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Track the loss and accuracy on the validation set to determine the best hyperparameters
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()

            with torch.no_grad():
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)
                    logps = model(images)
                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch + 1}/{epochs}.. "
                  f"Training loss: {running_loss / print_every:.3f}.. "
                  f"Validation loss: {test_loss / len(validloader):.3f}.. "
                  f"Test accuracy: {accuracy / len(validloader):.3f}")

            running_loss = 0
            model.train()

# Specifications:
# - The parameters of the feedforward classifier are appropriately trained, while the parameters of the feature network are left static
# - During training, the validation loss and accuracy are displayed


# TODO: Save the checkpoint
model.class_to_idx = train_data.class_to_idx

torch.save({'arch': arch,
            'inputs': inputs,
            'outputs': outputs,
            'state_dict': model.state_dict(),
            'epochs': epochs,
            'hidden_units': hidden_units,
            'classifier': model.classifier,
            'optimizer': optimizer.state_dict(),
            'class_to_idx': model.class_to_idx},
             checkpath)

# Specifications: The trained model is saved as a checkpoint along with associated hyperparameters and the class_to_idx dictionary