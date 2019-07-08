import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import os

class flower_model():
    def __init__(self, arch='resnet50',save_dir=os.path.dirname(os.path.abspath(__file__)), device='cpu',
                 learning_rate=0.03, hidden_units=512):
        # Load inputs into self
        self.save_dir = save_dir
        self.save_file = os.path.join(self.save_dir, 'checkpoint.pth')
        self.device = device
        # Initialize based on architecture
        if arch == 'densenet121':
            # Load the densenet121 pretrained model
            self.model = models.densenet121(pretrained=True)
            # Freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False
            # Replace the classifier
            self.model.classifier = nn.Sequential(nn.Linear(1024, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),
                                 nn.Linear(hidden_units, 102),
                                 nn.LogSoftmax(dim=1))
            self.criterion = nn.NLLLoss()
            self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=learning_rate)
        # Same steps as above but with resnet50 pretrained model
        elif arch == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.fc = nn.Sequential(nn.Linear(2048, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),
                                 nn.Linear(hidden_units, 102),
                                 nn.LogSoftmax(dim=1))
            self.criterion = nn.NLLLoss()
            self.optimizer = optim.Adam(self.model.fc.parameters(), lr=learning_rate)
        else:
            # Catch invalid models
            raise ValueError("Invalid Model Type")
        
    def save(self, train_dataset):
        # Moving the model to the cpu with the intent to initially save as cpu for easy loading and then move to as GPU as required; 
        self.model.to('cpu') 
        # Making the checkpoint (dictionary) that will contain everything needed to rebuild the model
        self.checkpoint = {'model_arch': self.model,
                      'state_dict': self.model.state_dict(),
                      'class_to_idx': train_dataset.class_to_idx}
        torch.save(self.checkpoint, self.save_file)
        
    def load(self, ckpt=None):
        # Setup to load with gpu or cpu based on input
        # Note: Not sure if this will work if trained on CPU and moved to GPU (not going to test CPU train time...)
        if ckpt == None:
            if self.device=='cuda':
                # If trained on GPU, can directly load 
                checkpoint = torch.load(self.save_file)
            else:
                # CPU load from GPU tensors found at https://discuss.pytorch.org/t/loading-weights-for-cpu-model-while-trained-on-gpu/1032 by user: fmassa
                checkpoint = torch.load(self.save_file, map_location=lambda storage, loc: storage)
        elif ckpt != None:
            if self.device=='cuda':
                checkpoint = torch.load(ckpt)
            else:
                checkpoint = torch.load(ckpt, map_location=lambda storage, loc: storage)
        # 
        self.model = checkpoint['model_arch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.class_to_idx = checkpoint['class_to_idx']
        #self.optimizer = optim.Adam(self.model.fc.parameters(), lr=0.003)
        #self.optimizer.state_dict = checkpoint['optim-state_dict']
        #return self.model, self.optimizer
        
    def self_test(self, testloader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        accuracy = 0
        test_loss = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    logps = self.model.forward(inputs)
                    batch_loss = self.criterion(logps, labels)
                    test_loss += batch_loss.item()
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            print("Test Loss: {:.3f}".format(test_loss/len(testloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
            model.train();
            model.to("cpu")
            