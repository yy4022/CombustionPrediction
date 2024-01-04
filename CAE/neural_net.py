from math import ceil

from torch import nn
from torchvision import transforms

"""
This file is used for building the Convolutional AutoEncoder neural network.
"""

class Encoder(nn.Module):

    def __init__(self, img_scale):
        super().__init__()

        self.img_scale = img_scale
        self.current_size = ceil((((((img_scale - 5)/4 + 1) - 3)/2 + 1) - 2)/2 + 1)

        # Convolutional section -- 3 layers
        self.encoder_cnn = nn.Sequential(
            # 1st Convolutional layer
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(negative_slope=0.3),
            # 1st MaxPooling layer
            nn.MaxPool2d(kernel_size=5, stride=4, padding=0, ceil_mode=True),

            # 2nd Convolutional layer
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(negative_slope=0.3),
            # 2nd MaxPooling layer
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True),

            # 3rd Convolutional layer
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(negative_slope=0.3),
            # 3rd MaxPooling layer
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
        )

        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.LeakyReLU(negative_slope=0.3),
        )

        # Linear section -- 3 layers
        self.encoder_lin = nn.Sequential(
            # 1st fully connected layer
            nn.Linear(in_features=128 * self.current_size * self.current_size,
                      out_features=128 * self.current_size),
            nn.LeakyReLU(negative_slope=0.3),

            # 2nd fully connected layer
            nn.Linear(in_features=128 * self.current_size, out_features=128),
            nn.LeakyReLU(negative_slope=0.3),

            # 3rd fully connected layer
            nn.Linear(in_features=128, out_features=100),
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):

    def __init__(self, img_scale):
        super().__init__()

        self.img_scale = img_scale
        self.current_size = ceil((((((img_scale - 5) / 4 + 1) - 3) / 2 + 1) - 2) / 2 + 1)

        # Linear section -- 3 layers
        self.decoder_lin = nn.Sequential(
            # 1st fully connected layer
            nn.Linear(in_features=100, out_features=128),
            nn.LeakyReLU(negative_slope=0.3),

            # 2nd fully connected layer
            nn.Linear(in_features=128, out_features=32 * self.current_size),
            nn.LeakyReLU(negative_slope=0.3),

            # 3rd fully connected layer
            nn.Linear(in_features=32 * self.current_size,
                      out_features=32 * self.current_size * self.current_size),
            nn.LeakyReLU(negative_slope=0.3),
        )

        # Unflatten section
        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(32, self.current_size, self.current_size))

        # Convolutional section -- 4 layers
        self.decoder_conv = nn.Sequential(
            # 1st Convolutional layer
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.3),
            # 1st Upsampling layer
            nn.Upsample(scale_factor=2, mode='bilinear'),

            # 2nd Convolutional layer
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.3),
            # 2nd Upsampling layer
            nn.Upsample(scale_factor=2, mode='bilinear'),

            # 3rd Convolutional layer
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.3),
            # 3rd Upsampling layer
            nn.Upsample(scale_factor=4, mode='bilinear'),

            # 4th Convolutional layer
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
        )

    # implement the decoder network in the given image
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        # crop the image size
        cropper = transforms.CenterCrop(size=(self.img_scale, self.img_scale))
        x = cropper(x)
        return x
