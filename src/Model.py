from __future__ import annotations
import utils
from typing import Any
import torch.nn as nn
import torch

class CustomEfficientNet(nn.Module):
    """CustomEfficientNet is pretrained EfficientNet module which weight is obtain from IMAGENET weights
    
    Parameters
    ----------
    
    num_of_target_label :  int
        number of class/target of used dataset (default: 1000)
    drop_out : float 
        probability value of nn.Dropout of its classifier (default: 0.4)
    device : torch.device 
       device available to locate the dataloader (default: torch.device('cpu'))
       
    Attributes
    ----------
    
    model : torchvision.models.EfficientNet
        hold EfficientNet instance from torchvision library
    
    Example
    import Models
    
    model = Models.CustomEfficientNet(num_of_target=3, drop_out = 0.4) 
    """

    def __init__(self, 
                 num_of_class: int = 1000, 
                 drop_out: float = 0.4, 
                 device: torch.device = torch.device('cpu')) -> None:
        
        super().__init__()
        
        self._tf = utils.get_EfficientNet_transform()       
        self.model = utils.get_EfficientNet(device=device)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=drop_out, inplace=True),
            nn.Linear(1280, num_of_class, bias=True)
        ).to(device=device)
        
    
    def _freeze(self) -> None:
        """provide helper for freezing all gradient (requires_grad = False) of features (extractor) layers"""
        for param in self.model.features.parameters():
            param.requires_grad = False
        
    
    def _unfreeze(self) -> None:
        """provide helper for unfreezing all gradient (requires_grad = True) of  features (extractor) layers"""
        for param in self.model.features.parameters():
            param.requires_grad = True
        
            
    
    def freeze(self) -> None:
        """freeze all gradient (requires_grad = False) of features (extractor) layers
        Parameters
        ----------
        -
        
        Example
        -------
        self.freeze()"""
        self._freeze()
        
    def unfreeze(self) -> None:
        """unfreeze all gradient (requires_grad = True) of features (extractor) layers
        Parameters
        ----------
        -
        
        Example
        -------
        self.unfreeze()"""
        self._unfreeze()
        
    def predict(self, X) -> None:
        X = self._tf(X)
        self.eval()
        X = self.model(X)
        probs = nn.Softmax(dim=1)(X)
        confidence, y_pred = torch.max(probs, dim=1)
        return confidence.item(), y_pred.item()
        
        
    def forward(self, x) -> Any:
        return self.model(x)
    
