from __future__ import annotations
import utils
import torch
from torchvision import transforms
from typing import Tuple, Dict
from CheckPoint import CheckPoint
from Config import Config
from EarlyStopping import EarlyStopping
from Plotting import Plotting
from Model import CustomEfficientNet

TRAIN_ACC = 'train_acc'
TRAIN_COST = 'train_cost'
TEST_ACC = 'test_acc'
TEST_COST = 'test_cost'
RED = '\033[91m'
GREEN = '\033[92m'
WHITE = '\033[97m'
BLACK = '\033[39m'

class Controller:
    """`Controller` instance act like an agent which controll all the training-loop related process 
    for example run the training and testing, provide the api to user for epoch mode or early stopping mode, 
    provide api to user for using adaption or fine-tunning phase in transfer learning and all internal related proccess in training loop
    
    Parameters
    ----------
    
    model : torch.nn.Module
        model used in for training-loop
    train_loader : torch.utils.data.DataLoader
        dataloader of training datasets used in training loop
    test_loader : torch.utils.data.DataLoader
        dataloder of testing datasets used in training loop
    criterion : torch.nn.Module
        loss function used in training loop
    optimizer : torch.optim.Optimizer
        optimzer used in training loop
    check_point : CheckPoint
        Checkpoint instance for storing some value produce in training loop
    config : Config
        Config instance used to get some configurable values used in training loop or in initiation
    early_stopping : EarlyStopping
        EarlyStopping instance used  to provide early stopping mode
    plotting : Plotting
        Plotting instance used  to provide plotting proccess
    device : torch.device
        device available to locate dataloader (default: torch.device('cpu'))
    epochs : int
        max epochs used in epochs mode or early stopping mode if needed
    
    Attributes
    ----------
    model, train_loader, test_loader, criterion, optimizer, cnf (config), cp (checkpoint)
    es (early_stopping), plt (plotting), max_epochs (epochs), device
    
    epoch : int
        used as epcohs counter (default: 1)
    ckpt : dict
        used to store value like accuary and cost of training step and testing step while running training loop
    result : list
        arra used to store all accyary and cost value of training step and testing step while running training loop
    
    Example
    -------
    # importing all library needed
    from Controller import Controller
    
    # initialize all object needed
    
    ctr = Controller(model=model,
                 test_loader=test_loader,
                 train_loader=train_loader,
                 criterion=criterion,
                 optimizer=optimizer,
                 writer=writer,
                 epochs=3,
                 check_point=cp,
                 early_stopping=ea,
                 plotting=plotting,
                 config=cnf,
                 device=torch.device('cpu'))

    """
    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 check_point: CheckPoint,
                 config: Config,
                 early_stopping: EarlyStopping,
                 plotting: Plotting,
                 device: torch.device = torch.device('cpu'),
                 epochs: int = 30
                 ) -> None:

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.cnf = config
        self.cp = check_point
        self.es = early_stopping
        self.plt = plotting
        self.epoch = 1
        self.max_epochs = epochs
        self.device = device
        self._run_fn = self._run_with_epoch
        self.ckpt = {
            TRAIN_ACC: 0,
            TRAIN_COST: 0,
            TEST_ACC: 0,
            TEST_COST: 0
        }
        self.result = {
            TRAIN_ACC: [],
            TRAIN_COST: [],
            TEST_ACC: [],
            TEST_COST: []
        }

    def _reset_ckpt(self) -> None:
        """reset a value  of ckpt attributes to its default value (zero value to all key)"""
        self.ckpt = {
            TRAIN_ACC: 0,
            TRAIN_COST: 0,
            TEST_ACC: 0,
            TEST_COST: 0
        }

    def _reset_result(self) -> None:
        """reset a value  of result attributes to its default value (empty list value to all key)"""
        self.result = {
            TRAIN_ACC: [],
            TRAIN_COST: [],
            TEST_ACC: [],
            TEST_COST: []
        }

    def _loop_fn(self, mode: str = 'train') -> Tuple[float, float]:
        """run test step or training step based on mode given in argument and return a tuple of accuary, cost value of its mode step
        
        Parameters
        ----------
        
        mode : str
            value which decide what step to run per batch wheterh is test step or training step  
        """

        dataloader = self.train_loader
        if mode == 'test':
            dataloader = self.test_loader
        
        loss = acc = cost = y_pred_class = 0
        for _, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(self.device), y.to(self.device)
            
            if mode == 'train':
                self.optimizer.zero_grad()
            
            # 1. Forward pass
            y_pred: torch.Tensor = self.model(X)
            
            # 2. Calculate  and accumulate loss
            loss = self.criterion(y_pred, y)
            if mode == 'train':
                loss.backward()

            # 3 backpropogation
            if mode == 'train':
                self.optimizer.step()
            
            # accumulate the accuracy and loss item
            y_pred_class = y_pred.argmax(dim=1)
            acc += torch.sum(y_pred_class == y).item() 
            cost += loss.item()

        return acc, cost

    def _add_ckpt_to_result(self):
        """append every ckpt value in dict to every result key"""
        self.result[TRAIN_ACC].append(self.ckpt[TRAIN_ACC])
        self.result[TRAIN_COST].append(self.ckpt[TRAIN_COST])
        self.result[TEST_ACC].append(self.ckpt[TEST_ACC])
        self.result[TEST_COST].append(self.ckpt[TEST_COST])

    def _inc_epochs(self) -> None:
        """increment epoch by 1 after all batch are trained (or equal 1 training loop)"""
        self.epoch += 1

    def _run_with_early_stopping(self, ckpt: CheckPoint, plt: Plotting, es: EarlyStopping, cnf: Config):
        """run training loop in early stopping mode
        
        Parameters
        ----------
        
        ckpt : CheckPoint
            CheckPoint instance used to log the model and save the weights of model to .pth file,
            also used to save the best weight of model while traning loop process
        plt : Plotting
            Plotting instance used to plot the accuary and cost of the model either for every certain epoch or after the training loop end
        es : EarlyStopping
            EarlyStopping instance used to provide logic of early stopping mode related
        cnf : Config
            Config instance used to provide cofigured value in json config file"""

        prev_acc_cost = {}
        while True:
            self._train_step()
            self._test_step()
            ckpt.log(train_acc=self.ckpt[TRAIN_ACC],
                     train_cost=self.ckpt[TRAIN_COST],
                     test_acc=self.ckpt[TEST_ACC],
                     test_cost=self.ckpt[TEST_COST])

            # DONE (checkpoint)
            plt.add_epoch_to_tick(self.epoch)
            ckpt.save_weights(cnf, self.epoch)
            # DONE (early stopping)
            # DONE (plotting)
            # plt.plot_acc_runtime(self.epoch, ckpt=ckpt)
            # plt.plot_cost_runtime(self.epoch, ckpt=ckpt)
            # self._report_per_epoch()
            self._write_report(self.epoch, prev_acc_cost)
            prev_acc_cost = self._set_acc_cost(prev_acc_cost)

            if es.is_stop(self.model, ckpt=ckpt, counter=self.epoch):
                self._run_fn = self._run_with_epoch
                plt.plot_acc(ckpt=ckpt)
                plt.plot_cost(ckpt=ckpt)
                break

            
            self._add_ckpt_to_result()
            self._reset_ckpt()
            self._inc_epochs()
    
    def _coloring_report(self, prev, key) -> str:
        """color the report line. red if the result lower the previous, green if the report higher, if there are no difference the color is white
        this function need to be refactored"""
        line = ''
        if prev[key] < self.ckpt[key]:
            line += (" | " + f"{key}: {GREEN}{self.ckpt[key]:.4f} ⬆ {BLACK}")
            return line
        elif prev[key] > self.ckpt[key]:
            line += (" | " + f"{key}: {RED}{self.ckpt[key]:.4f} ⬇ {BLACK}")
            return line
        else:
            line += (" | " + f"{key}: {self.ckpt[key]:.4f}   ")
            return line
    
    def _make_report_line(self, report : Dict, epoch: int) -> str:
        # line = f'\rEpoch: {self.epoch}/{self.max_epochs} '
        line = f'Epoch: {self.epoch}/{self.es.max_counter}'
        
        if epoch == 1 or not bool(report): # check if epoch 0 or report dict is empty
            line += (" | " + f"{TEST_ACC}: {self.ckpt[TEST_ACC]:.4f}   ")
            line += (" | " + f"{TEST_COST}: {self.ckpt[TEST_COST]:.4f}   ")
            line += (" | " + f"{TRAIN_ACC}: {self.ckpt[TRAIN_ACC]:.4f}   ")
            line += (" | " + f"{TRAIN_COST}: {self.ckpt[TRAIN_COST]:.4f}  ")
        else:
            line += self._coloring_report(report, TEST_ACC)
            line +=  self._coloring_report(report, TEST_COST)
            line +=  self._coloring_report(report, TRAIN_ACC)
            line +=  self._coloring_report(report, TRAIN_COST)
        return line
    
    def _set_acc_cost(self, report: Dict) -> Dict:
        report[TEST_ACC] = self.ckpt[TEST_ACC]
        report[TEST_COST] = self.ckpt[TEST_COST]
        report[TRAIN_ACC] = self.ckpt[TRAIN_ACC]
        report[TRAIN_COST] =  self.ckpt[TRAIN_COST]
        return report
    
    
    def _write_report(self, epoch: int, report) -> None:
        line = self._make_report_line(report=report, epoch=epoch)
        print(line, end='\n')
        
            
        
    def _run_with_epoch(self, plt: Plotting, ckpt: CheckPoint, mode : str = 'train_acc'):
        """provide logic training loop with epoch mode
        
        Paramters
        ---------
        plt : Plotting
            Plotting instance used to plot the accuary and cost of the model either for every certain epoch or after the training loop end
        ckpt : CheckPoint
            CheckPoint instance used to log the model and save the weights of model to .pth file,
            also used to save the best weight of model while traning loop process
        
        """
        prev_acc_cost = {}
        for _ in range(self.max_epochs):
            self._train_step()
            self._test_step()
            ckpt.log(test_acc=self.ckpt[TEST_ACC],
                     test_cost=self.ckpt[TEST_COST],
                     train_acc=self.ckpt[TRAIN_ACC],
                     train_cost=self.ckpt[TRAIN_COST])
            # print(f"\rEpoch: {self.epoch}/{self.max_epochs} test_acc: {self.ckpt[TEST_ACC]:.4f} | test_cost: {self.ckpt[TEST_COST]:.4f} | train_acc: {self.ckpt[TRAIN_ACC]:.4f} | train_cost: {self.ckpt[TRAIN_COST]:.4f}", end=' ')
            # prev = self._report_per_epoch(prev=prev)
            
            # self._report_result()
            self._write_report(self.epoch)
            prev_acc_cost = self._set_acc_cost(prev_acc_cost)
            plt.add_epoch_to_tick(self.epoch)
            self._add_ckpt_to_result()
            
            if self._is_improve(mode=mode, ckpt=ckpt):
                ckpt.weight = self.model.state_dict().copy()
                torch.save(ckpt.weights, self.cnf.MODEL_OUTPUT_DIR / "best_weights.pth")
                
            
            self._inc_epochs()
        
        plt.plot_acc(ckpt=ckpt)
        plt.plot_cost(ckpt=ckpt)
    
    def _is_improve(self, ckpt: CheckPoint, mode : str = 'test_acc') -> bool:
        """Return true if model is improve else return false
        Improved in here mean the monitored mode value is higher than the past
        
        Paramaters
        ----------
        
        ckpt : CheckPoint
            checkpoint instnace which hold the reference accuracy/cost and current max accuracy or min cost\
        mode : str
            mode to be referencing to monitor model for deciding whether the model improved or not
            for example if mode `test_acc` so only the accuracy of testing to be monitored for deciding if model improved or not
            
        Example
        -------
    
        self._is_improved(ckpt) # assume ckpt is valid argument"""
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
        
        if improve:
            if mode.endswith('_acc'):
                ckpt.best_acc = ref
            elif mode.endswith('_cost'):
                ckpt.best_cost = ref
            return True
        return False

    def _train_step(self) -> None:
        """Provide logic for training step process. will create side effect (set value (accuracy and cost) to ckpt attributes)"""
        self.model.train()
        acc, cost = self._loop_fn(mode='train')
        self.ckpt[TRAIN_ACC] = acc / len(self.train_loader.dataset)
        self.ckpt[TRAIN_COST] = cost / len(self.train_loader.dataset)

    def _test_step(self) -> None:
        """Provide logic for testing step process. will create side effect (set value (accuracy and cost) to ckpt attributes)"""
        self.model.eval()
        with torch.inference_mode():
            acc, cost = self._loop_fn(mode='test')
            self.ckpt[TEST_ACC] = acc / len(self.test_loader.dataset)
            self.ckpt[TEST_COST] = cost / len(self.test_loader.dataset)
        

    def adaptation(self) -> Controller:
        """Provide API for adaptation phase by freezing the extractor layers of model"""
        self.model.freeze()
        return self

    def fine_tunning(self) -> Controller:
        """Provide API for fine-tuning phase by unfreezing the extractor layers of model"""
        self.model.unfreeze()
        return self
    
    def with_epoch(self)-> Controller:
        """Provide API for set controller to run in epoch mode """
        self._run_fn = lambda : self._run_with_epoch(plt=self.plt, ckpt=self.cp)
        return self

    def with_early_stopping(self) -> Controller:
        """Provide API for set controller to run in early stopping mode """
        self._run_fn = lambda: self._run_with_early_stopping(
            ckpt=self.cp, plt=self.plt, es=self.es, cnf=self.cnf)
        return self

    def run(self):
        """this function will run the whole training loop either by epochs or early stopping (another mode could be added before like adaptation or fine-tunning)
        
        Parameter
        ---------
        -
        
        Example
        -------
        ctrl = Controller(*args) # assume this is valid for sake of simplicity
        
        # only call these methods should be doing the whole training loop
        ctrl.adaptation().with_early_stopping().run()
        """
        self._reset_result()
        self._reset_ckpt()
        self._run_fn()
        self.epoch = 1
        
    @staticmethod
    def new_from_config(cnf : Config, augmentation_func  : transforms.Compose = None) -> Controller:
        """create a `Controller` instance using json config file
        
        Parameters
        ----------
        
        cnf : Config
            Config instance which hold all configured value
        augmentation_func : torchvision.Composes
            transfomation function used to augmenting datasets
            
        Example
        -------
        from Config import Config
        
        Config.new_from_config('path/to/config.json', augment_transform)"""
        device = utils.get_device()
        transfom_func = utils.get_EfficientNet_transform()
        trainloader, testloader, _ = utils.create_data_loader(
            train_path=cnf.TRAIN_PATH, test_path=cnf.TEST_PATH,
            transform_func=transfom_func, batch_size=cnf.BATCH_SIZE,
            augmentation_func=augmentation_func)        
        model = CustomEfficientNet(num_of_class=cnf.NUM_OF_CLASS, drop_out=cnf.DROP_OUT, device=device)
        es = EarlyStopping(cnf=cnf, max_counter=cnf.ES_EPOCHS, load_best_when_stop=cnf.LOAD_BEST, patience=cnf.PATIENCE)
        cp = CheckPoint()
        plt = Plotting(plot_every=cnf.PLOT_EVERY)
        
        return Controller(
            model=model,
            test_loader=testloader,
            train_loader=trainloader,
            optimizer=utils.get_optimizer(name=cnf.OPTIMIZER_NAME, lr=cnf.LEARNING_RATE, weight_decay=cnf.WEIGHT_DECAY),
            criterion=torch.nn.CrossEntropyLoss(),
            early_stopping=es,
            check_point=cp,
            plotting=plt,
            epochs=cnf.EPOCHS,
            device=device
        )
