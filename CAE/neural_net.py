from math import ceil

from torch import nn
from torchvision import transforms

"""
This file is used for building the Convolutional AutoEncoder neural network.
"""

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        # Convolutional section
        self.encoder_cnn = nn.Sequential(
            # 1st Convolutional layer
            nn.Conv2d(1, 32, 5, padding=2),
            nn.LeakyReLU(negative_slope=0.3),
            # 1st MaxPooling layer
            nn.MaxPool2d(kernel_size=(2, 2), ceil_mode=True),
            # 2nd Convolutional layer
            nn.Conv2d(32, 64, 5, padding=2),
            nn.LeakyReLU(negative_slope=0.3),
            # 2nd MaxPooling layer
            nn.MaxPool2d(kernel_size=(2, 2), ceil_mode=True),
            # 3rd Convolutional layer
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.3),
            # 3rd MaxPooling layer
            nn.MaxPool2d(kernel_size=(2, 2), ceil_mode=True),
        )

        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.LeakyReLU(negative_slope=0.3),
        )

        # Linear section
        self.encoder_lin = nn.Sequential(
            # 1st fully connected layer
            nn.Linear(128 * ceil(self.img_scale / 8) * ceil(self.img_scale / 8),
                      8 * ceil(self.img_scale / 8) * ceil(self.img_scale / 8)),
            nn.LeakyReLU(negative_slope=0.3),
            # 2nd fully connected layer
            nn.Linear(8 * ceil(self.img_scale / 8) * ceil(self.img_scale / 8), 64),
            nn.LeakyReLU(negative_slope=0.3),
            # 3rd fully connected layer
            nn.Linear(64, 20),
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

class Decoder(nn.Module):

    def __init__(self, img_scale):
        super().__init__()

        # Linear section
        self.decoder_lin = nn.Sequential(
            # 1st fully connected layer
            nn.Linear(20, 32),
            nn.LeakyReLU(negative_slope=0.3),
            # 2nd fully connected layer
            nn.Linear(32, 64),
            nn.LeakyReLU(negative_slope=0.3),
            # 3rd fully connected layer
            nn.Linear(64, 8 * ceil(self.img_scale / 8) * ceil(self.img_scale / 8)),
            nn.LeakyReLU(negative_slope=0.3),
        )

        # Unflatten section
        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(8, ceil(self.img_scale / 8), ceil(self.img_scale / 8)))

        # Convolutional section
        self.decoder_conv = nn.Sequential(
            # 1st Convolutional layer
            nn.Conv2d(8, 128, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.3),
            # 1st Upsampling layer
            nn.Upsample(scale_factor=2, mode='bilinear'),
            # 2nd Convolutional layer
            nn.Conv2d(128, 64, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.3),
            # 2nd Upsampling layer
            nn.Upsample(scale_factor=2, mode='bilinear'),
            # 3rd Convolutional layer
            nn.Conv2d(64, 32, 5, padding=2),
            nn.LeakyReLU(negative_slope=0.3),
            # 3rd Upsampling layer
            nn.Upsample(scale_factor=2, mode='bilinear'),
            # 4th Convolutional layer
            nn.Conv2d(32, 1, 5, padding=2),
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        # crop the image size
        cropper = transforms.CenterCrop(size=(self.img_scale, self.img_scale))
        x = cropper(x)
        return x
