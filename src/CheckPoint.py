import torch
from typing import List, Dict, Any, Union
import numpy as np
from Config import Config



class CheckPoint:
    """
    `CheckPoint` create an instance which store some value like `train_acc`, `test_cost`, model `weights` and more. 
    these values used by other objects like `EarlyStopping` or `Plotting`. this object provide function logging and save weight for some epoch in .pth file.
    `Controller` instance is dependend on this class through depedency injection.
    
    Parameters
    ----------
    -
    
    Attributes
    ----------
    
    train_acc : List of float or List of Tensor
        all train score/accurary along the training loop
    train_cost : List of float or List of Tensor
        all train cost along the training loop
    test_acc : List of float or List of Tensor
        all test score/accurary along the training loop
    test_cost : List of float or List of Tensor
        all test cost along the training loop
    weights:
        the state dict of model or nn.Module, used to store best weights if early stopping mode is on (this decide by user through Controller API)
    best_acc : int
        max value of train_acc while training and test_acc while testing
    best_cost: int
        min value of train_cost while training and test_cost while testing
    save_every: int
        constant which indicate when to save model's weights per certain epoch
    
    Example
    -------
    from Checkpoint import Checkpoint
    
    Checkpoint()
    """
    def __init__(self) -> None:
        self.train_acc: List = []
        self.train_cost: List = []
        self.test_acc: List = []
        self.test_cost: List = []
        self.weights: Dict = {}
        self.best_acc: int = 0
        self.best_cost: int = np.inf
        self.save_every: int = 50
    
    def save_weights(self, cnf: Config, epoch: int)-> None:
        """Save the weights of the model for every cofigured epoch
        
        Parameters
        ----------
        
        cnf : Config
            config object for providing target output directory/path
        epoch : int
            current epoch, if epoch match or multiple of configured epoch the weights is saved
        
        Example
        --------
        cp = CheckPoint()
        
        cp.save_weights(cnf, epoch)
 
        """
        if epoch % self.save_every == 0:
            torch.save(self.weights, cnf.MODEL_OUTPUT_DIR / f"weights_{epoch}.pth")
    
    def log(self, 
            train_acc: Union[float, torch.Tensor, None] = None, 
            train_cost: Union[float, torch.Tensor, None] = None, 
            test_acc: Any = Union[float, torch.Tensor, None], 
            test_cost: Any = Union[float, torch.Tensor, None]) -> None:
        """Will log some values and set those value to CheckPoint instance
        
        Parameters
        ----------
        
        train_acc : float, torch.Tensor or None
            training accuary value
        train_cost:  float, torch.Tensor or None
            training cost value
        test_acc : float, torch.Tensor or None
            test accuary value
        test_cost:  float, torch.Tensor or None
            test cost value
        
        Example
        -------
        cp = Checkpoint()
        
        cp.log(train_acc, train_cost, test_acc, test_cost)
        """
        if train_acc is not None:
            train_acc = train_acc.item() if type(train_acc) == torch.Tensor else train_acc
            self.train_acc.append(train_acc)
        if train_cost is not None:
            train_cost = train_cost.item() if type(train_cost) == torch.Tensor else train_cost
            self.train_cost.append(train_cost)
        if test_acc is not None:
            test_acc = test_acc.item() if type(test_acc) == torch.Tensor else test_acc
            self.test_acc.append(test_acc)
        if test_cost is not None:
            test_cost = test_cost.item() if type(test_cost) == torch.Tensor else test_cost
            self.test_cost.append(test_cost)
            
    def reset(self) -> None:
        """Reset the attribues of self/instance
        
        Parameters
        ----------
        -
        
        Example
        -------
        cp = CheckPoint()
        
        cp.reset()
        """
        self.train_acc: List = []
        self.train_cost: List = []
        self.test_acc: List = []
        self.test_cost: List = []
        self.weights: Dict = {}
        self.best_acc: int = 0
        self.best_cost: int = np.inf
        self.save_every: int = 50
        