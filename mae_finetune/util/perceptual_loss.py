import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    """
    Computes perceptual loss using features extracted from a pre-trained VGG19 network.
    The loss measures the difference between the predicted and target images
    in a feature space defined by intermediate VGG19 layers.
    """

    def __init__(self, layer_ids=None, weight=0.1, device='cpu'):
        """
        Initializes the PerceptualLoss module.

        Args:
            layer_ids (list of ints, optional): Indices of VGG19 layers to use for feature extraction.
                                                Defaults to [9, 16], corresponding to relu2_2 and relu3_3.
            weight (float): Weight to scale the perceptual loss relative to other losses.
            device (str): Device to run the VGG network on ('cpu' or 'cuda').
        """
        super().__init__()
        
        # Default VGG19 layers for feature extraction: relu2_2 (layer 9) and relu3_3 (layer 16)
        if layer_ids is None:
            layer_ids = [9, 16]

        # Load the pre-trained VGG19 model and freeze its parameters
        vgg = models.vgg19(pretrained=True).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Create a list of slices of the VGG model up to the specified layers
        self.vgg_slices = nn.ModuleList()
        prev_j = 0
        for j in layer_ids:
            slice_j = nn.Sequential(*vgg[prev_j:j]).to(device)
            self.vgg_slices.append(slice_j)
            prev_j = j
        self.weight = weight

        # No gradients for VGG
        for slice_net in self.vgg_slices:
            for p in slice_net.parameters():
                p.requires_grad = False


    def forward(self, pred, target):
        """
        Computes the perceptual loss between the predicted and target images.

        Args:
            pred (torch.Tensor): Predicted image tensor with shape [B, 3, H, W], normalized with ImageNet mean/std.
            target (torch.Tensor): Target image tensor with shape [B, 3, H, W], normalized with ImageNet mean/std.

        Returns:
            torch.Tensor: The computed perceptual loss (scalar).
        """
        loss = 0.0
        x_pred = pred
        x_target = target

        # Pass the images through each VGG slice and compute the MSE loss in the feature space
        for slice_net in self.vgg_slices:
            x_pred = slice_net(x_pred)
            x_target = slice_net(x_target)
            loss += torch.mean((x_pred - x_target)**2)

        # Scale the loss by the defined weight
        return loss * self.weight
