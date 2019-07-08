import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import os


def build_dataloaders(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    train_transform = transforms.Compose([transforms.RandomRotation(35),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224,0.225])])
    # Validation transform to check the progress of the model while it trains. Images not augmented, only resized and cropped.
    valid_transform = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224,0.225])])
    # Test transform to check the model's effectiveness after it is done training. Images not augmented, only resized and cropped.
    test_transform = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224,0.225])])

    # TODO: Load the datasets with ImageFolder
    # Dataset to hold the images and apply the transforms
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    # Loaders to load the images in batches. 
    # The trainloader is shuffled to mix batches while training to prevent potential artifacts.
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)    
    
    return (train_dataset, valid_dataset, test_dataset), (trainloader, validloader, testloader)

def process_image(image):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    '''
    # Keeping imports in the function for now
    import torch
    import numpy as np
    from PIL import Image
    # Open the image
    im = Image.open(image)
    # Scale the image dependent on the shortest side to keep the aspect ratio
    if im.size[0] < im.size[1]:
        im.thumbnail((256,1000))
    elif im.size[1] < im.size[0]:
        im.thumbnail((1000,256))
    # Cropping the center of the image
    im = im.crop(box=(im.size[0]/2-112,im.size[1]/2-112,im.size[0]/2+112,im.size[1]/2+112))
    # Normalization mean and standard deviation
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Taking the numpy array from the image
    np_image = np.array(im)
    # Normalizing the image
    np_image = ((np_image / 255 - mean) / std)
    # Transpose
    np_image = np_image.transpose(2,0,1)
    # Convert back to a tensor for the model
    im_out = torch.from_numpy(np_image)
    # Converting to TensorFloat (previously TensorDouble)
    im_out = im_out.float()
    # Reshaping to be used in the model (Batch_Size, Color_Channels, Pixel_Y(?), Pixel_X(?))
    im_out = im_out.reshape((1,3,224,224))
    return im_out

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    # I Intend to put imports in beginning in command line code, helps me to leave them here for this notebook
    import numpy as np
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.reshape((3,224,224))
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.set_title(title)
    ax.imshow(image)
    
    return ax

