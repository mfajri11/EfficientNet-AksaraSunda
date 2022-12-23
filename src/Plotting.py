from CheckPoint import CheckPoint
from typing import Sequence, Union, List
import matplotlib.pyplot as plt

class Plotting:
    """`Plotting` instance provide helper for plotting the accuracy and cost while training loop or at the end of training loop
    
    Parameter
    ---------
    
    plot_every : int
        constant which decide when to plot per it value epochs.
        example plot_every is 10, so every 10 epoch and if wont plot per epoch set its value to zero (0)
    
    Attributes
    ----------
    plot_every
    
    plot_tick : List[int]
        a list which hold range of epochs value when the plotting process run
        
    Example
    -------
    from Plotting import Plotting
    
    plt = Plotting(plot_every=10)"""
    def __init__(self, plot_every: int) -> None:
        self.plot_every = plot_every
        self.plot_tick: List[int] = list()
    
    # this function should be share accross the instance and not isolated for each instance
    # thus make it static should be good choice
        
    
    @staticmethod
    def _plot_func(scale: str = 'linear') -> Union[Union[plt.plot, plt.semilogy], None]:
        """static factory function which return plt.plot or plt.semilogy function
        the condition describe below
        scale == 'linear' -> plt.plot
        scale == 'semilogy' -> plt.semilogy
        
        Paramtere
        ---------
        
        scale : str
            scale mode which which plot function should be returned (default: 'linear')
        """
        if scale == 'linear':
            plot_func = plt.plot
        elif scale == 'semilogy':
            plot_func = plt.semilogy
        else:
            raise Exception("Plotting only support scale mode {'linear', 'semilogy'}")
        return plot_func
    
    def add_epoch_to_tick(self, epoch: int) -> None:
        self.plot_tick.append(epoch)
    
    def _plot(self, 
              mode:str, 
              scale: str,
              figzie: Sequence[float],
              ckpt: CheckPoint) -> None:
        """helper which actually perform plotting proccess
        
        Parameters
        ----------
        
        mode : str
            indicate what value want to plot either accuracy or cost (possible arguments are 'acc' or 'cost' )
        scale : str
            scale mode of plot if 'linear' plot function used is plt.semilogy (default: 'semilogy')
        figsize : Sequence[float]
            the size of plot figure (default: (8,5))
        ckpt : Checkpoint
            Checkpoint instance which provide value needed to plot
        
        Example
        -------
        plt = Plotting(10)
        # assume all arguments are valid
        
        plt._plot(mode='acc', scale=scale, figzie=figsize, ckpt=ckpt)
        """
        plot_func = self._plot_func(scale=scale)
        
        plt.figure(figsize=figzie)
        plt.ylabel(mode.title())
        plt.xlabel('Epoch')
        mode = mode.lower()
        if mode == 'acc':
            # print('train_acc: ', ckpt.train_acc)
            # print('tick: ', self.plot_tick)
            plot_func(self.plot_tick, ckpt.train_acc, 'r-', label='Train')
            if len(ckpt.test_acc):
                # print('test_acc: ', ckpt.test_acc)
                plot_func(self.plot_tick, ckpt.test_acc, 'b-', label='Test')
        elif mode == 'cost':
            plot_func(self.plot_tick, ckpt.train_cost, 'r-', label='Train')
            if len(ckpt.test_cost):
                plot_func(self.plot_tick, ckpt.test_cost, 'b-', label='Test')
        plt.legend()
        plt.show()
        
    def reset_tick(self):
        """reset plot_tick attributes to empty list"""
        self.plot_tick = list()
    
    def plot_acc(self, 
                 ckpt: CheckPoint,
                 figsize: Sequence[float] = (8,5), 
                 scale: str = 'linear') -> None:
        """plot the accuracy of training/test step at the end of training loop using plt.plot function
        
        Parameters
        ----------
        
        ckpt : Checkpoint
            Checkpoint instance which provide value needed to plot
        figsize : Sequence[float]
            the size of plot figure (default: (8,5))
        scale : str
            scale mode of plot if 'linear' plot function used is plt.plot (default: 'linear')
        
        Example
        -------
        plt = Plotting(10)
        # assume ckpt is valid argument
        
        plt.plot_acc(ckpt=ckpt)
        """
        self._plot(mode='acc', scale=scale, figzie=figsize, ckpt=ckpt)
    
    def plot_cost(self,
                  ckpt: CheckPoint,
                  figsize: Sequence[float] = (8,5), 
                  scale: str = 'semilogy') -> None:
        """plot the cost of training/test step at the end of training loop using plt.semilogy function 
        
        Parameters
        ----------
        
        ckpt : Checkpoint
            Checkpoint instance which provide value needed to plot
        figsize : Sequence[float]
            the size of plot figure (default: (8,5))
        scale : str
            scale mode of plot if 'linear' plot function used is plt.semilogy (default: 'semilogy')
        
        Example
        -------
        plt = Plotting(10)
        # assume ckpt is valid argument
        
        plt.plot_cost(ckpt=ckpt)"""
        self._plot(mode='cost', scale=scale, figzie=figsize, ckpt=ckpt)
    
    def plot_acc_runtime(self,
                         epoch: int,
                         ckpt: CheckPoint,
                         figsize: Sequence[float] = (8,5),
                         scale: str = 'linear') -> None:
        """plot the accuracy of training/test step every specified epoch using plt.plot function
        
        Parameters
        ----------
        
        epoch : int
            number which decide when to plot while runtime / training loop running and it will plot per its epoch number 
        ckpt : Checkpoint
            Checkpoint instance which provide value needed to plot
        figsize : Sequence[float]
            the size of plot figure (default: (8,5))
        scale : str
            scale mode of plot if 'linear' plot function used is plt.plot (default: 'linear')
            
        Example
        -------
        plt = Plotting(10)
        # assume epoch and ckpt are all valid arguments
        
        plt.plot_acc_runtime(epoch, ckpt)"""
        
        if self.plot_every % epoch == 0 and epoch > 0:
            self._plot(mode='linear', figzie=figsize,scale=scale, ckpt=ckpt)
    
    def plot_cost_runtime(self,
                         epoch: int,
                         ckpt: CheckPoint,
                         figsize: Sequence[float] = (8,5),
                         scale: str = 'semilogy') -> None:
        """plot the accuracy of training/test step every specified epoch using plt.semilogy function
        
        Parameters
        ----------
        
        epoch : int
            number which decide when to plot while runtime / training loop running and it will plot per its epoch number 
        ckpt : Checkpoint
            Checkpoint instance which provide value needed to plot
        figsize : Sequence[float]
            the size of plot figure (default: (8,5))
        scale : str
            scale mode of plot if 'linear' plot function used is plt.plot (default: 'semilogy')
            
        Example
        -------
        plt = Plotting(10)
        # assume epoch and ckpt are all valid arguments
        
        plt.plot_cost_runtime(epoch, ckpt)"""
        if self.plot_every % epoch == 0 and epoch > 0:
            self._plot(mode='semilogy', figzie=figsize,scale=scale, ckpt=ckpt)
            