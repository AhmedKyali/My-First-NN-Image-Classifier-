#import some important files
import argparse
import json
import PIL
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from torchvision import datasets, transforms, models
import torchvision.models as models
import torchvision
from collections import OrderedDict
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import train

#define functions
def process_image(image_path):
    #Process to use in model
    img_pil = Image.open(image_path)
    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    image = img_transforms(img_pil)

    return image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
        
    image = image.numpy().transpose((1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # clip image between 0 and 1 
    image = np.clip(image, 0, 1)
    ax.imshow(image)

    return ax
#predict the class from an image file
def predict(image_path, model, number_of_outputs=5):
    #convert to cuda or cpu depends on user
    model.to(device)
    model.eval()
    img = process_image(image_path).numpy()
    img = torch.from_numpy(np.array([img])).float()

    with torch.no_grad():
        logps = model.forward(img.cuda())

    ps = torch.exp(logps).data
    #probs contains the highest probapilities and classes has the label of it
    probs, classes = ps.topk(args.top_k, dim=1)
    #convert tensor to numpy array
    probs = probs.cpu().detach().numpy().tolist()[0]
    classes = classes.cpu().detach().numpy().tolist()[0]
    #get names of the class by its index
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_class = [idx_to_class[i] for i in classes]
    classes = np.array(top_class)
    classes = classes.astype(np.int)
    return probs, classes



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', default='flowers/test/29/image_04137.jpg', action="store", type = str)

    parser.add_argument('--checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str)

    parser.add_argument('--top_k', dest="top_k", action="store", type=int, default=5)

    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')

    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")
    args = parser.parse_args()
    image_path = args.input
    number_of_outputs = args.top_k
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu =='gpu' else 'cpu')
    json_name = args.category_names
    path = args.checkpoint
    
    with open(json_name, 'r') as f:
        cat_to_name = json.load(f)
    
    checkpoint = torch.load(path)
    checkpoint
    model = getattr(models, checkpoint['structure'])(pretrained=True)
    model.to(device)
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    epochs = checkpoint["epochs"]

    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    #Display an image
    plt.rcdefaults()
    fig, ax = plt.subplots()

    index = 1
    path = image_path
    probs, classes = predict(image_path, model,number_of_outputs)

    
    image = process_image(path)

    ax1 = imshow(image, ax = plt)
    ax1.axis('off')
    
    a = np.array(probs)
    b = [cat_to_name[str(index)] for index in classes]
    
    fig,ax2 = plt.subplots(figsize=(number_of_outputs,number_of_outputs))


    y_pos = np.arange(number_of_outputs)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(b)
    ax2.set_xlabel('Probability')
    ax2.invert_yaxis()
    ax2.barh(y_pos, a)

    plt.show()