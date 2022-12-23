import torch
from CheckPoint import CheckPoint
from Config import Config


_RED = '\033[91m'
_BLACK = '\033[39m'

class EarlyStopping:
    """create `EarlyStopping` instance which hold logic to provide early stopping mode in training loop
    
    Parameters
    ----------
    
    cnf : Config
        CheckPoint instance which provide value like list of test accuracy, best accuacy, and also used to store the best weight got in training loop\
    max_counter : int
        maximum number of epoch this needed for interupt purpose in case trainig proccess are too heavy to run (default: 0)
    load_best_when_stop : bool
        if true will be loaded best wheight to the trained model if false the last model will be used by the trained model (default: true)
    patience : int
        teshold for unimproved model (acc/cost lower or same for certain number of times) (default: 5)
        
    Attributes
    ----------
    
    cnf, max_counter, early_stop_patience, stop, load_best_when_stop
    
    stop : bool
        if true will stop the training loop iteration if false the training loop will be continue
    
    Example
    --------
    fron EarlyStooping import EarlyStopping
    
    es = EarlyStopping(cnf, max_counter=100, load_best_when_stop=true, patience=5) # assume all arguments are valid 
        """
    def __init__(self,
                 cnf: Config,
                 max_counter: int = 30,
                 load_best_when_stop: bool = True,
                 patience: int = 5) -> None:
        self.cnf = cnf
        self.max_counter = max_counter
        self._early_stop_counter = 0
        self.early_stop_patience = patience
        self.stop = False
        self.load_best_when_stop = load_best_when_stop
        self._98 = 0
    
    def _save_weight(self, ckpt: CheckPoint) -> None:
        """Save the best weight based on accuary score in training loop
        
        Paramter
        --------
        ckpt : CheckPoint
            `Checkpoint` instance provide best weight got in training loop in
            
        Example
        -------
        es = EarlyStopping(*args) # assume this valid this actually not valid initialization
        
        self._save_weight(ckpt=ckpt) # assume ckpt is valid argument"""
        
        # check if directory already exists or not
        if not self.cnf.MODEL_OUTPUT_DIR.exists():
            self.cnf.MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        #save weights (pth)
        torch.save(ckpt.weights, self.cnf.MODEL_OUTPUT_DIR / f"weight_1e6_7030.pth")
        
    
    def reset_early_stop(self) -> None:
        """Reset `early_stop_counter` attribute to 0"""
        self._early_stop_counter = 0
    
    def is_stop(self,
                model: torch.nn.Module,
                ckpt: CheckPoint,
                counter: int,
                device: str = 'cpu',
                mode: str = 'test_acc') -> bool:
        """return true if early stopping process is done
        
        Parameters
        ----------
        
        model : torch.nn.Module
            model used to save its best weight while training and loaded when the training loop is done
        ckpt : CheckPoint
            used to provide all the value needed to decide weither the model improve or not and to store the best weight and best accuracy and cost
        counter : int
            current running epoch number
        mode : str
            variable to control which used to tell if the model improve could be test accuary, trainig accuary, test cost or training cost (default: 'test_acc')
        device : torch.device
            to locate dataloader when the best model loaded (default: torch.device('cpu'))
        
        Example
        -------
        es = EarlyStopping(*args) # assume this valid this actually not valid initialization
        
        es.is_stop(model, ckpt, counter, mode='test_cost') # assume all arguments are valid
        """
        mode = mode.lower()
        improve = False
        stop = False
        if mode == 'test_acc':
            ref = ckpt.test_acc[-1]
            improve = ref > ckpt.best_acc
        elif mode == 'test_cost':
            ref = ckpt.test_cost[-1]
            improve = ref < ckpt.best_cost
        elif mode == 'train_acc':
            ref = ckpt.train_acc[-1]
            improve = ref > ckpt.best_acc
        elif mode == 'train_cost':
            improve = ckpt.best_cost
        else:
            raise Exception(f"can't support mode: {mode}, modes available are {'test_acc', 'test_cost', 'train_acc', 'train_cost'}")     

        if ref >= 0.98:
          self._98 += 1

        best = ckpt.best_acc if mode.endswith("_acc")  else ckpt.best_cost
        if improve:
            if mode.endswith('_acc'):
                ckpt.best_acc = ref
                best = ckpt.best_acc
            else:
                ckpt.best_cost = ref
                best = ckpt.best_cost
            
            ckpt.weights = model.state_dict().copy()
            self._save_weight(ckpt=ckpt)
            
            if self._early_stop_counter > 0:
              self.reset_early_stop()
              print(f"{_BLACK}Model improve, EarlyStop patience reset to 0{_BLACK}")
            
        else:
            self._early_stop_counter += 1
            print(f"{_RED}EarlyStop Patience: {self._early_stop_counter} | best {mode}: {best:.4f}{_BLACK}")

            if self._early_stop_counter >= self.early_stop_patience:
              stop = True
              print(f"EarlyStop stop at {self._early_stop_counter} iteration on epoch {counter}")

        if counter >= self.max_counter  or self._98 > 3 or ckpt.best_acc > 0.99: # or ckpt.best_acc > 0.98:
            print(f"EarlyStop stop at iteration on epoch {counter}")
            stop = True
          
        if stop:
          print(f"Best {mode}: {best:.4f}")
          print(f"Best model saved at {self.cnf.MODEL_OUTPUT_DIR / 'weights_best.pth'}")
          if self.load_best_when_stop:
              weights = torch.load(self.cnf.MODEL_OUTPUT_DIR / "weight_1e6_7030.pth",
                                    map_location=device)
              model.load_state_dict(weights)
        
        return stop
    
    def set_patience(self, num: int) -> None:
        """set num to `early_stop_patience` attributes
        
        Paramter
        --------
        
        num : int
            value to set to `early_stop_patience`
        
        Example
        -------
        es = EarlyStopping(*args) # assume this valid this actually not valid initialization
        
        es.set_patience(3)"""
        self.early_stop_patience = num