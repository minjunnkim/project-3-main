import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        """
        super(SimpleNet, self).__init__()

        self.conv_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()
        self.loss_criterion = None

        ############################################################################
        # Student code begin
        ############################################################################

        # Convolutional + Pooling layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, padding=2),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                                 

            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, padding=2), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                                
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),                         
            nn.Linear(20 * 16 * 16, 128),          
            nn.ReLU(),
            nn.Linear(128, 15)                     
        )

        # Loss function
        self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')

        ############################################################################
        # Student code end
        ############################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the net

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        """
        model_output = None
        ############################################################################
        # Student code begin
        ############################################################################
        
        model_output = self.conv_layers(x)
        model_output = self.fc_layers(model_output)

        ############################################################################
        # Student code end
        ############################################################################

        return model_output
