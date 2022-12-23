from __future__ import annotations
import pathlib
import json
import utils
from typing import Any, Union
from pathlib import Path
import numpy as np

class Config:
    """Config class provide a class which convert json object of config file to python object
    
    Parameters
    ----------
    learning_rate : float
        learning-rate value for Optimizer
    epochs : int
        max epoch of training loop (include  training and testing)
    drop_out : float
        probability value of nn.Dropout of classifier's CustomefficientNet
    batch_size: int
        batchsize value for dataloader
    es_epochs: int
        max epochs if needed while running model in early stopping mode, for purpose resource efficiency e.g. while running on local
    patience: int
        treshold which decided stop training loop proccess in early stopping mode
    plot_every: int
        plot acc/cost of trainig/testing for every certain number and its multiple
    ratio_split : str
        train and testing ratio for splitting datasets
    weight_decay : float
        constant for weight-decay's torch optimzer
    load_best : bool
        if true when early stopping stop `Controller` will load the best weights of training model
    num_of_class : int
        number of class/target of dataset used
    dataset_name : str
        the name of folder/directory of dataset used
    optimizer_name : str
        the name of used optimizer
    base_output_dir : str, pathlike
        the base directory of output file like, model's weight (.pth) or saved json config file
    model_output_dir : str, pathlike
        the directory of model related output
    base_data_dir : str, pathlike
        the base directory of used dataset
    
    Attributes
    ----------
    LEARNING_RATE, DROP_OUT, EPOCHS, BATCH_SIZE, ES_EPOCHS, PATIENCE, PLOT_EVERY, RATIO_SPLIT, 
    WEIGHT_DECAY, LOAD_BEST, NUM_OF_CLASS, DATASET_NAME, OPTIMIZER_NAME, BASE_OUTPUT_DIR, MODEL_OUTPUT_DIR, BASE_DATA_DIR
    
    Example
    -------
    this class should not be initialize/construct direclty, recommend to use `load_config` function
    
    """
    def __init__(self,
                 learning_rate: float,
                 epochs: int,
                 drop_out: float,
                 batch_size: int,
                 es_epochs: int,
                 patience: int,
                 plot_every: int,
                 ratio_split: str,
                 weight_decay: float,
                 load_best: bool,
                 num_of_class: int,
                 dataset_name: str,
                 optimizer_name: str,
                 base_output_dir: Union[str, pathlib.Path] = '../output',
                 model_output_dir: Union[str, pathlib.Path] =  "../output/model",
                 base_data_dir: Union[str, pathlib.Path] = '../data') -> None:
        
        self.BASE_OUTPUT_DIR = utils.convert_str_to_path(base_output_dir)
        self.MODEL_OUTPUT_DIR = utils.convert_str_to_path(model_output_dir)
        self.BASE_DATA_DIR = utils.convert_str_to_path(base_data_dir)
        self.LEARNING_RATE = learning_rate
        self.DATASET_NAME = dataset_name
        self.EPOCHS = epochs
        self.DROP_OUT = drop_out
        self.BATCH_SIZE = batch_size
        self.ES_EPOCHS = es_epochs
        self.PATIENCE = patience
        self.PLOT_EVERY = plot_every
        self.RATIO_SPLIT = ratio_split
        self.WEIGHT_DECAY = weight_decay
        self.LOAD_BEST = load_best
        self.OPTIMIZER_NAME = optimizer_name
        self.NUM_OF_CLASS = num_of_class
        self.TRAIN_PATH = self.BASE_DATA_DIR /self.DATASET_NAME / 'train'
        self.TEST_PATH = self.BASE_DATA_DIR / self.DATASET_NAME / 'test'
        
    def get_dict(self):
        data = {}
        
        for k, v in self.__dict__.items():
            if isinstance(v, pathlib.Path):
                v = str(v)
            data[k.lower()] = v
            
        return data
        
    def __str__(self) -> str:
        """Return string representation of instance
        Parameters
        ----------
        -
        
        Example
        -------
        Import Config
        cnf = Config.load_config('path/to/config.json')
        
        cnf.__str__() or str(cnf) or print(cnf)
        """
        return str(self.__dict__)
    

    
def load_config(path: Union[str, pathlib.Path] ='../config/config.json') -> Config:
    """
    Return Config object from given config json file
    
    Parameters
    ----------
    
    path : str, pathlike
        path or location of config file
    
    Example
    -------
    import Config
    
    Config.load_config('path/to/config.json')
    """
    path = utils.convert_str_to_path(path=path)
    if not path.is_file:
        raise Exception(f"the path file : {path} doesn't exist")
    with open(path) as f:
        data = json.load(f)
        cnf: Config = Config(**data)
    return cnf

def generate_all_possible_config(target, lrs=[], eps=[], is_include_es = True, opts=[], ratios=[]) -> None:
    def lr_to_str(x):
        if x == 1e-1:
            return '1e-1'
        elif x == 1e-2:
            return '1e-2'
        elif x == 1e-3:
            return '1e-3'
        elif x == 1e-4:
            return '1e-4'
        elif x == 1e-5:
            return '1e-5'
        else:
            return str(x)
    lrs_str = list(map(lambda x :f"lr{lr_to_str(x)}", lrs))
    eps_str = list(map(lambda x: f"ep{x}", eps))
    cnf : Config = Config.load_config()

    
    cnfs_name = []
    for _, ratio in enumerate(ratios):
        for _, opt in enumerate(opts):
            for iep, ep in enumerate(eps_str):
                for ilr, lr in enumerate(lrs_str):
                    cnf.LEARNING_RATE = lrs[ilr]
                    cnf.OPTIMIZER_NAME = opt
                    cnf.EPOCHS = eps[iep]
                    cnf.RATIO_SPLIT = ratio
                    
                    rt = ratio.replace('/', '-')
                    cnf_name = f"{opt}_{ep}_{lr}_{rt}"
                    cnfs_name.append(cnf_name)
                    
                    data = cnf.get_dict()
                    with open(target+"/"+cnf_name+".json", mode='w') as f:
                        data = json.dumps(data, indent=4)
                        f.write(data)
            if is_include_es:
                for ilr, lr in enumerate(lrs_str):
                    cnf.LEARNING_RATE = lrs[ilr]
                    cnf.OPTIMIZER_NAME = opt
                    cnf.RATIO_SPLIT = ratio
                    cnf.EPOCHS = 0
                    
                    rt = ratio.replace('/', '-')
                    cnf_name = f"{opt}_es_{lr}_{rt}"
                    cnfs_name.append(cnf_name)
                    
                    data = cnf.get_dict()
                    with open(target+"/"+cnf_name+".json", mode='w') as f:
                        data = json.dumps(data, indent=4)
                        f.write(data)
    return cnfs_name