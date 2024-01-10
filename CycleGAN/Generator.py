import torch
import torch.nn as nn
import torch
import torch.nn as nn

# CNN block is used for downsampling and upsampling the input 
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", **kwargs):
        super(CNNBlock, self).__init__()
        
        conv = []
        if down:
            conv.append(nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs))
        else:
            conv.append(nn.ConvTranspose2d(in_channels, out_channels, **kwargs))
        conv.append(nn.InstanceNorm2d(out_channels))
    
        if  act == "relu" :
            conv.append(nn.ReLU(inplace=True))
        else:
            conv.append(nn.Identity())
            
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            CNNBlock(channels, channels, kernel_size=3, padding=1),
            CNNBlock(channels, channels, act="identity", kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super(Generator, self).__init__()
        
        # Initial block
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True)
        )
        
        # Downsample blocks
        down_blocks = []
        down_blocks.append(CNNBlock(num_features, num_features * 2, kernel_size=3, stride=2, padding=1))
        down_blocks.append(CNNBlock(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)
        
        # Residual blocks
        residual_blocks = []
        for _ in range(num_residuals):
            residual_blocks.append(ResidualBlock(num_features * 4))
        self.res_blocks = nn.Sequential(*residual_blocks)
        
        # Upsample blocks
        up_blocks = []
        up_blocks.append(CNNBlock(num_features * 4, num_features * 2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1))
        up_blocks.append(CNNBlock(num_features * 2, num_features * 1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.up_blocks = nn.ModuleList(up_blocks)
        
        # Last block
        self.last = nn.Sequential(
            nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        
        for layer in self.down_blocks:
            x = layer(x)
            
        x = self.res_blocks(x)
        
        for layer in self.up_blocks:
            x = layer(x)
            
        return self.last(x)

