# My First NN Image Classifier

## Overview

This project is part of Udacity's AI Programming with Python Nanodegree program. It involves developing an image classifier using PyTorch, and converting it into a command line application. The project demonstrates how to train a deep learning model on a dataset of images, save the model, and then use it to make predictions on new images.

## Features

- Implementation of an image classifier using PyTorch
- Training the model with a dataset of flower images
- Saving and loading the trained model
- Predicting image classes using a trained model
- Command line interface for training and prediction

## Files in the Repository

- `Image Classifier Project.ipynb`: Jupyter notebook with the initial development and testing of the image classifier.
- `LICENSE`: License file for the project.
- `README.md`: This file.
- `cat_to_name.json`: JSON file mapping category labels to flower names.
- `predict.py`: Script for making predictions with the trained model.
- `running_train_and_predict.ipynb`: Jupyter notebook demonstrating how to run the training and prediction scripts.
- `test.jpg`: Example image for testing predictions.
- `train.py`: Script for training the image classifier.
- `workspace-utils.py`: Utility functions for workspace management.

## Installation

To run this project, you need to have Python installed along with the following libraries:

- NumPy
- Matplotlib
- PyTorch
- torchvision
- PIL

You can install these libraries using pip:

```sh
pip install numpy matplotlib torch torchvision pillow
```

## Usage

### Training the Model

To train the model, use the `train.py` script. You can specify various parameters such as the dataset directory, save directory, model architecture, learning rate, hidden units, number of epochs, and whether to use GPU. Here is an example command:

```sh
python train.py --data_dir ./flowers --save_dir ./checkpoint.pth --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 3 --gpu gpu
```

### Predicting with the Model

To make predictions with the trained model, use the `predict.py` script. You can specify the image file, checkpoint file, top K classes, category names file, and whether to use GPU. Here is an example command:

```sh
python predict.py --input flowers/test/1/image_06754.jpg --checkpoint ./checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu gpu
```

## Project Structure

- `train.py`: Script for training the image classifier.
- `predict.py`: Script for making predictions with the trained model.
- `cat_to_name.json`: JSON file mapping category labels to flower names.
- `running_train_and_predict.ipynb`: Jupyter notebook demonstrating how to run the training and prediction scripts.
- `Image Classifier Project.ipynb`: Jupyter notebook with the initial development and testing of the image classifier.
- `test.jpg`: Example image for testing predictions.
- `workspace-utils.py`: Utility functions for workspace management.

## Example Code

### Training Script (`train.py`)

```python
# Imports here
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
import json
from collections import OrderedDict
from torch.utils import data
import argparse

# Build network
def network(structure, lr, hidden_units):
    if structure == "vgg11":
        model = models.vgg11(pretrained=True)
    elif structure == "vgg13":
        model = models.vgg13(pretrained=True)
    elif structure == "vgg16":
        model = models.vgg16(pretrained=True)
    elif structure == "vgg19":
        model = models.vgg19(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, 256)),
        ('relu', nn.ReLU()),
        ('fc3', nn.Linear(256, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model = model.to(device)

    optimizer = optim.Adam(model.classifier.parameters(), lr)

    criterion = nn.NLLLoss()

    return model, criterion, optimizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', action="store", default="./flowers")
    parser.add_argument('--save_dir', action="store", default="./checkpoint.pth")
    parser.add_argument('--arch', action="store", default="vgg16")
    parser.add_argument('--learning_rate', action="store", type=float, default=0.001)
    parser.add_argument('--hidden_units', action="store", type=int, default=512)
    parser.add_argument('--epochs', action="store", default=3, type=int)
    parser.add_argument('--dropout', action="store", type=float, default=0.2)
    parser.add_argument('--gpu', action="store", default="gpu")

    args = parser.parse_args()
    where = args.data_dir
    path = args.save_dir
    lr = args.learning_rate
    structure = args.arch
    hidden_units = args.hidden_units
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu == 'gpu' else 'cpu')
    epochs = args.epochs
    dropout = args.dropout

    data_dir = where
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_data = ImageFolder(root=train_dir, transform=train_transforms)
    valid_data = ImageFolder(root=valid_dir, transform=test_transform)
    test_data = ImageFolder(root=test_dir, transform=test_transform)

    train_loader = data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = data.DataLoader(valid_data, batch_size=64)
    test_loader = data.DataLoader(test_data, batch_size=64)

    model, criterion, optimizer = network(structure, lr, hidden_units)

    steps = 0
    print_every = 5

    for e in range(epochs):
        running_loss = 0
        for inputs, labels in train_loader:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        log_ps = model.forward(inputs)
                        batch_loss = criterion(log_ps, labels)
                        valid_loss += batch_loss.item()

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {e+1}/{epochs}.. "
                      f"Loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {valid_loss/len(valid_loader):.3f}.. "
                      f"Accuracy: {accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train()

    model.eval()
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        final_accuracy = accuracy / len(test_loader)
        print("Test Accuracy: {:.3f}".format(final_accuracy))

    model.class_to_idx = train_data.class_to_idx
    torch.save({
        'input_n': 25088,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optim': optimizer.state_dict(),
        'output_n': 102,
        'lr': lr,
        'classifier': model.classifier,
        'epochs': epochs,
        'structure': structure
    }, 'checkpoint.pth')
    print("checkpoint saved!")
```

### Prediction Script (`predict.py`)

```python
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision import transforms,

 models
from PIL import Image
import json
import argparse

def process_image(image):
    transformers = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    processed_img = transformers(image)
    return processed_img

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    image = image.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def predict(image_path, model, topk=5):
    model.to(device)
    image = Image.open(image_path)
    processed_img = process_image(image)
    processed_img = processed_img.unsqueeze_(0)
    processed_img = processed_img.float()

    with torch.no_grad():
        out = model.forward(processed_img.cuda())

    probs = F.softmax(out.data, dim=1)
    top_probs = probs.topk(topk)[0][0]

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [np.int(idx_to_class[each]) for each in np.array(probs.topk(topk)[1][0])]

    return top_probs, top_classes

def check(path, model):
    probs, classes = predict(image_path, model)
    image = process_image(Image.open(image_path))
    plt.subplot(211)
    ax_1 = imshow(image, ax=plt)
    ax_1.axis('off')
    top_class = classes[0]
    ax_1.title(cat_to_name[str(top_class)])
    ax_1.show()

    labels = [cat_to_name[str(index)] for index in classes]
    y_pos = np.arange(len(labels))
    plt.barh(y_pos, probs, align='center')
    plt.yticks(y_pos, labels)
    plt.gca().invert_yaxis()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', default='flowers/test/1/image_06754.jpg', action="store", type=str)
    parser.add_argument('--checkpoint', default='./checkpoint.pth', nargs='?', action="store", type=str)
    parser.add_argument('--top_k', dest="top_k", action="store", type=int, default=5)
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

    args = parser.parse_args()
    image_path = args.input
    number_of_outputs = args.top_k
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu == 'gpu' else 'cpu')
    json_name = args.category_names
    path = args.checkpoint

    with open(json_name, 'r') as f:
        cat_to_name = json.load(f)

    checkpoint = torch.load('checkpoint.pth')
    model = getattr(models, checkpoint['structure'])(pretrained=True)
    model.to('cuda')
    model.classifier = checkpoint['classifier']
    model.input_size = checkpoint['input_n']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['lr'])
    optimizer.load_state_dict(checkpoint['optim'])
    model.output_size = checkpoint['output_n']
    model.learning_rate = checkpoint['lr']
    model.epochs = checkpoint['epochs']

    check(image_path, model)
```

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.


## Contact

For any questions or suggestions, feel free to open an issue or contact me at ahmed.kaiialy@gmail.com.
