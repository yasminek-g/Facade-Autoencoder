import torch.nn as nn

class RefinementNet(nn.Module):
    """
    A simple 2-layer Convolutional Neural Network (CNN) designed for refining image predictions.
    The network takes an image as input and outputs a refined version of the image.
    """

    def __init__(self):
        """
        Initializes the RefinementNet with a 2-layer CNN:
        - First convolutional layer with 64 output channels and ReLU activation.
        - Second convolutional layer that outputs 3 channels (same as input).
        """
        super().__init__()

        # Define a sequential 2-layer CNN
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor with shape [B, 3, H, W], where
                              B = Batch size, 3 = Number of channels (RGB), H = Height, W = Width.

        Returns:
            torch.Tensor: Refined image tensor with the same shape [B, 3, H, W].
        """

        return self.conv(x)
