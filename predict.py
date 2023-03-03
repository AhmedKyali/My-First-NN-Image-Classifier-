# Imports here
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from torchvision.datasets import ImageFolder
from PIL import Image
import json
from collections import OrderedDict 
import torchvision
from torch.utils import data
import argparse

#define images
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    transformers = transforms.Compose([transforms.Resize(256)
                                     ,transforms.CenterCrop(224)
                                     ,transforms.ToTensor()
                                     ,transforms.Normalize(
                                      mean=[0.485, 0.456, 0.406]
                                     ,std=[0.229, 0.224, 0.225])])
    processed_img = transformers(image)
    return processed_img


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax



def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    image=Image.open(image_path)
    #process the image to fit the model
    processed_img = process_image(image)
    processed_img = processed_img.unsqueeze_(0)
    processed_img = processed_img.float()
    #forward
    with torch.no_grad():
        out = model.forward(processed_img.cuda())
    #softmax activation function 
    probs = F.softmax(out.data,dim=1)
    top_probs = probs.topk(topk)[0][0]
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [np.int(idx_to_class[each]) for each in np.array(probs.topk(topk)[1][0])]
    
    return top_probs, top_classes



def check(path,model):
    probs, classes = predict(image_path, model)
    image = process_image(Image.open(image_path))
    plt.subplot(211)    
    ax_1 = imshow(image, ax = plt)
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

    parser.add_argument('--input', default='flowers/test/1/image_06754.jpg', action="store", type = str)

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
    
    
    #load the model
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

    check(image_path,model)