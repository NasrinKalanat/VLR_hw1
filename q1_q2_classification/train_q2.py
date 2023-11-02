import torch
import trainer
from utils import ARGS
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset
import numpy as np
import torchvision
import torch.nn as nn
import random


class ResNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        self.resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        ##################################################################
        # TODO: Define a FC layer here to process the features
        ##################################################################
        # for param in self.resnet.parameters():
        #     param.requires_grad=False
            
        num_feat=self.resnet.fc.in_features
        self.resnet.fc=nn.Sequential(
            nn.Linear(num_feat,256),
            nn.ReLU(),
        #    nn.Dropout(0.5),
            nn.Linear(256,num_classes)
        )
        #self.resnet.fc=nn.Linear(num_feat,num_classes)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
        

    def forward(self, x):
        ##################################################################
        # TODO: Return raw outputs here
        ##################################################################
        return self.resnet(x)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    ##################################################################
    # TODO: Create hyperparameter argument class
    # We will use a size of 224x224 for the rest of the questions. 
    # Note that you might have to change the augmentations
    # You should experiment and choose the correct hyperparameters
    # You should get a map of around 50 in 50 epochs
    ##################################################################
    for lr, batch_size,step_size,gamma in [(0.0001, 32, 5, 0.1)]:
        args = ARGS(
            epochs=200,
            inp_size=224,
            use_cuda=True,
            val_every=70,
            lr=lr,
            batch_size=batch_size,
            step_size=step_size,
            gamma=gamma
        )
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

        print(args)

        ##################################################################
        # TODO: Define a ResNet-18 model (https://arxiv.org/pdf/1512.03385.pdf) 
        # Initialize this model with ImageNet pre-trained weights
        # (except the last layer). You are free to use torchvision.models 
        ##################################################################

        model = ResNet(len(VOCDataset.CLASS_NAMES)).to(args.device)

        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

        # initializes Adam optimizer and simple StepLR scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        # trains model using your training code and reports test map
        test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
        print('test map:', test_map)
        if test_map>=0.8:
            print("************ ","lr: ",lr,", batch_size: ",batch_size,", step_size: ",step_size,", gamma: ", gamma)
