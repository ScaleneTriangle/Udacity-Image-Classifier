import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
from utility_functions import build_dataloaders 
from model import flower_model
import os
import argparse
from workspace_utils import keep_awake

def train(data_dir, save_dir=os.path.dirname(os.path.abspath(__file__)), arch='resnet50', learning_rate=0.003, hidden_units=512, epochs=3, device=None):
    if device:
        device='cuda'
    else:
        device='cpu'
        
    model = flower_model(save_dir=save_dir, arch=arch, learning_rate=learning_rate, hidden_units=hidden_units)
    ts, tl = build_dataloaders(data_dir)
    train_dataset = ts[0]
    trainloader, validloader, testloader = tl
          
    # Setup variables
    model.model.to(device)
    steps = 0  # Count steps for validation
    print_every = 3 # Number of steps until validation occurs
    running_loss = 0 # Accumulate train loss to be averaged

    # Training Loop
    for e in keep_awake(range(epochs)):
        # Loop for each epoch using the trainloader to output batches of images.
        for inputs, labels in trainloader:
            # Counting up steps until next validation
            steps += 1
            # Transfer the inputs and labels to the GPU (if active)
            inputs, labels = inputs.to(device), labels.to(device)
            # Zeroing the gradients of the optimizer, don't actually understand how this helps.
            # Assuming something to prevent the optimizer from adding gradients for each .step(). Not sure why not automatic...
            model.optimizer.zero_grad()
            # Running a forward pass. LogSoftmax outputs the logarithmic probabilites. Can retrieve probs with e^logps
            logps = model.model.forward(inputs)
            # Finding the loss of the output compared to the label
            loss = model.criterion(logps, labels)
            # Finding the gradients with back propagation
            loss.backward()
            # Apply the gradients to the model classifier (fc) parameters with the optimizer
            model.optimizer.step()

            # Update the running loss to be averaged in the validation step. Not really sure what the .item() is actually pulling...
            running_loss += loss.item()

            # Validation pass, every print_every number of steps
            if steps % print_every == 0:
                # Preventing the model from training, not exactly sure how this differs from torch.no_grad()...
                model.model.eval()
                # Initialize variables for validation
                accuracy = 0
                valid_loss = 0
                # Doing another type of preventing gradients from being built, same comment as model.eval() note...
                with torch.no_grad():
                    for inputs, labels in validloader:
                        # Same steps as in trainer, but without doing steps to find backprop gradients and update the model
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.model.forward(inputs)
                        batch_loss = model.criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        # Steps to find the accuracy of the model on the validation images
                        # probs = e^log(probs)
                        ps = torch.exp(logps)
                        # Finding the category with the highest probability
                        top_p, top_class = ps.topk(1, dim=1)
                        # Finding if the top_category matches the label (check if guess is correct). Not sure what the * is...
                        equals = top_class == labels.view(*top_class.shape)
                        # Finding the accuracy, don't entirely understand this step... How many were correct? .item? Why average in print?
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                # Format the print. Finding the averages of the accumulated losses. Not sure I understand accuracy (what is being accumulated?)
                print("Epoch {}/{}".format(e+1, epochs),
                      "Train Loss: {:.3f}".format(running_loss/print_every),
                      "Validation Loss: {:.3f}".format(valid_loss/len(testloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(testloader)))
                # Reset running_loss
                running_loss = 0
                # Set the model back to training for the next epoch
                model.model.train()
    return model
          
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training the Model")
    parser.add_argument("data_dir")
    parser.add_argument("--save_dir", action="store", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--arch", action="store", default="resnet50")
    parser.add_argument("--learning_rate", action="store", default=0.003, type=float)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--hidden_units", action="store", default=512, type=int)
    parser.add_argument("--epochs", action="store", default=3, type=int)
    args = parser.parse_args()
    model = train(args.data_dir, save_dir=args.save_dir, arch=args.arch, learning_rate=args.learning_rate, device=args.gpu, hidden_units=args.hidden_units, epochs=args.epochs)
    
    ts, tl = build_dataloaders(args.data_dir)
    train_dataset = ts[0]
    model.save(train_dataset)
    print("Training Complete. Checkpoint saved at {}".format(model.save_file))
