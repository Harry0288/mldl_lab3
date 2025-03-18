import torch
from torch import nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

x = torch.ones(1, device=device)
x

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Add more layers...
        # add another convolutional layer to extract deeper features
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        #self.fc1 = nn.Linear(..) # 200 is the number of classes in TinyImageNet
        # Fully connected layers for classification, as convolutional layers
        # are used just for extracting features from images, then you have to use
        # fully connected layers to make classification
        # self.fc1 = nn.Linear(200704, 512)  # Assuming input image size is 64x64 (Tiny ImageNet)
        # 200704 is too big, you either change size in convolutional layer, or in
        # pooling change the kernel to be bigger, for instance 5 instead of 2
        # self.fc2 = nn.Linear(512, 200)  # 200 classes in TinyImageNet

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 200)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 6)  # Reduce size by half, it is a good practice
        # to define self.maxpool = F.max_pool() and then in forward just call it
        # as otherwise you redefine the function each time if you leave it in forward
        # For relu, since it doesn't have any parameters except the input data,
        # you can leave it in forward
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 6)  # Reduce size by half
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 6)  # Reduce size by half

        # Flatten before passing to fully connected layers
        x = torch.flatten(x,1)  # Flatten to (batch_size, features)
        # print(x.shape)
        # x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Output layer (logits)
        # logits = F.relu(x)

        return x