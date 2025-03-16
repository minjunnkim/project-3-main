import torch
import torch.nn as nn


class SimpleNetFinal(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        """
        super(SimpleNetFinal, self).__init__()

        self.conv_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()
        self.loss_criterion = None

        ############################################################################
        # Student code begin
        ############################################################################

        # Convolutional + Pooling layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, padding=2),
            nn.BatchNorm2d(10),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   
            
            nn.Dropout(p=0.5),                              

            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, padding=2), 
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=20, out_channels=30, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                                
        ) 

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),                         
            nn.Linear(30 * 8 * 8, 128),          
            nn.ReLU(),
            nn.Linear(128, 15)                     
        )

        # Loss function
        self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')
        
        self._init_weights()

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
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)  
                nn.init.constant_(m.bias, 0)
