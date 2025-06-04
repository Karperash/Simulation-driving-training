import torch
import torch.nn as nn
import torch.nn.functional as F

class RGBBCNet(nn.Module):
    def __init__(self):
        super(RGBBCNet, self).__init__()
        
        # Convolutional layers for RGB image processing
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Calculate the size of flattened features
        self.feature_size = self._calculate_conv_output_size((224, 224))
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3)  # 3 outputs: steering, throttle, brake
        
        self.dropout = nn.Dropout(0.5)
    
    def _calculate_conv_output_size(self, input_size):
        # Helper function to calculate the size of the flattened features
        x = torch.zeros(1, 3, *input_size)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x.numel() // x.size(0)
    
    def forward(self, x):
        # Convolutional layers with ReLU activation and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Output processing
        steer = torch.tanh(x[:, 0])  # Normalize steering to [-1, 1]
        throttle = torch.sigmoid(x[:, 1])  # Normalize throttle to [0, 1]
        brake = torch.sigmoid(x[:, 2])  # Normalize brake to [0, 1]
        
        return steer, throttle, brake 