#import some important files
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image
import json
from collections import OrderedDict 
import torchvision


#define some function

#Build network
def network(structure , lr , hidden_units):
    # region build model using pretained vgg model    
    if structure == "vgg11":
        model = torchvision.models.vgg11(pretrained=True)
    elif structure == "vgg13":
        model = torchvision.models.vgg13(pretrained=True)
    elif structure == "vgg16":
        model = torchvision.models.vgg16(pretrained=True)
    elif structure == "vgg19":
        model = torchvision.models.vgg19(pretrained=True)

    # freeze model parameters
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
    #use argparse to take parameters from user in command lines
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', action="store", default="./flowers")
    parser.add_argument('--save_dir', action="store", default="./checkpoint.pth")
    
    parser.add_argument('--arch', action="store", default="vgg16")
    parser.add_argument('--learning_rate', action="store", type=float,default=0.001)
    parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=512)
    
    parser.add_argument('--epochs', action="store", default=3, type=int)
    parser.add_argument('--dropout', action="store", type=float, default=0.2)
    
    parser.add_argument('--gpu', action="store", default="gpu")
    #put the args in variables to make it easier below
    args = parser.parse_args()
    where = args.data_dir
    path = args.save_dir
    lr = args.learning_rate
    structure = args.arch
    hidden_units = args.hidden_units
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu =='gpu' else 'cpu')
    epochs = args.epochs
    dropout = args.dropout
    
    
    #load the data
    data_dir = where
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    #load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)

    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    
    #define the dataloaders
    dataloaders = trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)

    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = True)

    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64, shuffle = True)
    
    #Label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    
    
    model, criterion, optimizer = network(structure , lr , hidden_units)
    
    #train the model
    print_every = 5
    steps = 0
    loss_show = []

    for e in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for inputs, labels in validloader:
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
                      f"Validation Loss: {valid_loss/len(validloader):.3f}.. "
                      f"Accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
                
    #test model
    test_loss = 0
    accuracy = 0
    model.to(device)

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            log_ps = model.forward(inputs)
            batch_loss = criterion(log_ps, labels)

            test_loss += batch_loss.item()

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test accuracy: {accuracy/len(testloader):.3f}")
    
    #Save the checkpoint
    model.class_to_idx = train_data.class_to_idx
    torch.save({'input_size': 25088,
                'output_size': 102,
                'structure': structure,
                'learning_rate': lr,
                'classifier': model.classifier,
                'epochs': epochs,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx}, path)
    print("checkpoint saved, program finished")
    
