import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()

        if down:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect")
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x

class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(Generator, self).__init__()

        # Initial downsampling
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        # Downsampling
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False)
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False)
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        # Upsampling
        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False)
        self.up5 = Block(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False)
        self.up6 = Block(features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False)
        self.up7 = Block(features * 2 * 2, features    , down=False, act="relu", use_dropout=False)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )


    def forward(self, x):
        down1 = self.initial_down(x)
        down2 = self.down1(down1)
        down3 = self.down2(down2)
        down4 = self.down3(down3)
        down5 = self.down4(down4)
        down6 = self.down5(down5)
        down7 = self.down6(down6)
        bottleneck = self.bottleneck(down7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, down7], 1))
        up3 = self.up3(torch.cat([up2, down6], 1))
        up4 = self.up4(torch.cat([up3, down5], 1))
        up5 = self.up5(torch.cat([up4, down4], 1))
        up6 = self.up6(torch.cat([up5, down3], 1))
        up7 = self.up7(torch.cat([up6, down2], 1))
        
        return self.final_layer(torch.cat([up7, down1], 1))