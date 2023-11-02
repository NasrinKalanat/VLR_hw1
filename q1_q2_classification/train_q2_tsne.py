import torch
import trainer
from utils import ARGS
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset
import numpy as np
import torchvision
import torch.nn as nn
import random
import utils


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

def feature_extractor(args, model):
    test_loader = utils.get_data_loader(
        'voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)

    # Ensure model is in correct mode and on right device
    model.resnet=nn.Sequential(*list(model.resnet.children())[:-1])
    model.eval()
    model = model.to(args.device)

    cnt = 0
    feat=[]
    gt=[]
    print(len(test_loader))
    with torch.no_grad():
      for batch_idx, (data, target, wgt) in enumerate(test_loader):
          data, target, wgt = data.to(args.device), target.to(args.device), wgt.to(args.device)
          cnt+=data.shape[0]

          output = model(data)
          feat.append(output.detach().cpu().numpy())
          gt.append(target.detach().cpu().numpy())
          if cnt>=1000:
            return np.array(feat), np.array(gt)


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
            epochs=50,
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
        feat, gt = feature_extractor(args, model)
        feat=feat.squeeze()
        gt=gt.squeeze()

        print('test map:', test_map)
        if test_map>=0.8:
            print("************ ","lr: ",lr,", batch_size: ",batch_size,", step_size: ",step_size,", gamma: ", gamma)

        from sklearn.manifold import TSNE
        tsne=TSNE(n_components=2)
        feat=tsne.fit_transform(feat)
        print(feat.shape)

        def help(x):
            return np.mean(np.nonzero(x))

        classes=np.apply_along_axis(help, 1, gt)
        classes=classes.astype(np.int32)

        from matplotlib.legend_handler import Line2D
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.set_palette("Paired")
        cls_name=[VOCDataset.CLASS_NAMES[c.item()] for c in classes]
        df=pd.DataFrame({"t-SNE Component 1":feat[:,0], "t-SNE Component 2":feat[:,1], "class":cls_name})
        _,ax=plt.subplots(1)
        sns.scatterplot(x="t-SNE Component 1", y="t-SNE Component 2", hue="class", data=df, ax=ax)
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.subplots_adjust(right=0.7)
        plt.savefig("tsne.png")