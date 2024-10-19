import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.residual_downsample = None
        if stride != 1 or in_channels != out_channels:
            self.residual_downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))

    
    def forward(self, x):
        if self.residual_downsample:
            residual = self.residual_downsample(x)
        else:
            residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x = self.relu(x + residual)
        return x

class ResNetLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        super().__init__(*layers)

class ResNetModel(nn.Module):
    def __init__(self, num_channels=[64, 128, 256, 512], num_blocks=[2, 2, 2, 2], num_classes=10):
        super().__init__()
        assert len(num_channels) == 4
        assert len(num_blocks) == 4

        self.conv = nn.Conv2d(3, num_channels[0], kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(num_channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layers = [ResNetLayer(num_channels[0], num_channels[0], num_blocks[0], stride=1)]
        for i in range(1, 4):
            layers.append(ResNetLayer(num_channels[i - 1], num_channels[i], num_blocks[i], stride=2))
        self.resnet_layers = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels[-1], num_classes)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)

        x = self.resnet_layers(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# model = ResNetModel()
# inputs = torch.randn(16, 3, 224, 224)
# outputs = model(inputs)
# print(outputs.shape)