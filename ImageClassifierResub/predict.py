import argparse
import json
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
from utility_functions import build_dataloaders, process_image, imshow
from model import flower_model
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
    
    
def predict(image_path, checkpoint_path="checkpoint.pth", top_k=1, category_names=None, device=None):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if device:
        device='cuda'
    else:
        device='cpu'
        
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    model = flower_model()
    model.load(checkpoint_path)
    model.model.to(device)
    # Processing image and setting to GPU if active
    img = process_image(image_path)
    img = img.to(device)
    #img = img.float()
    # Running a forward step with the model to classify the image
    model.model.eval()
    with torch.no_grad():
        logps = model.model.forward(img)
        ps = torch.exp(logps)
        # .topk finds the highest probabilities classes of the model outputs
        top_p, top_class = ps.topk(top_k)
    top_class = np.array(top_class.reshape(-1))
    top_p = np.array(top_p.reshape(-1))
    class_name = []
    for k in top_class:
        for i in model.model.class_to_idx.keys():
            if model.model.class_to_idx[str(i)] == k:
                class_name.append(cat_to_name[i])
    return top_p, class_name    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training the Model")
    # flowers/test/1/image_06743.jpg
    # checkpoint.pth
    parser.add_argument("image_path")
    parser.add_argument("checkpoint_path")
    parser.add_argument("--top_k", action="store", default=2, type=int)
    parser.add_argument("--category_names", action="store", default='cat_to_name.json', type=str)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
                        
    top_p, class_name = predict(args.image_path, checkpoint_path=args.checkpoint_path, top_k=args.top_k, category_names=args.category_names, device=args.gpu)  
    print('Classes      : {}'.format(class_name))
    print('Probabilities: {}'.format(top_p))
    print('Figure saved to Prediction.png')
    im = process_image(args.image_path)
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    imshow(im, ax=ax1, title=class_name[0])
    ax2.barh([i+1 for i in range(len(class_name))][::-1], width=top_p, tick_label=class_name)
    plt.tight_layout()
    ax2.set_aspect(0.06)
    fig.set_size_inches(6,6)
    fig.savefig('Prediction.png')
    plt.show()
                
    