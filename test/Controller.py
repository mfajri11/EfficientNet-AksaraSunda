import unittest
from unittest import mock
from unittest.mock import MagicMock, Mock, patch


from src.Controller import Controller

class TestController(unittest.TestCase):
    
    @patch('torch.device')
    @patch('src.Plotting.Plotting')
    @patch('src.EarlyStopping.EarlyStopping')
    @patch('src.Config.Config')
    @patch('src.CheckPoint.CheckPoint')
    @patch('torch.optim.Optimizer')
    @patch('torch.nn.Module')
    @patch('torchvision.datasets.ImageFolder')
    @patch('torchvision.datasets.ImageFolder')
    @patch('src.Model.CustomEfficientNet')
    def setUp(self,
              mock_eff,
              mock_trainloader,
              mock_testloader,
              mock_criterion,
              mock_optimizer,
              mock_cp,
              mock_cnf,
              mock_es,
              mock_plt,
              mock_device) -> None:
        
        self.ctrl = Controller(
            model=mock_eff,
            test_loader=mock_testloader,
            train_loader=mock_trainloader,
            criterion=mock_criterion,
            optimizer=mock_optimizer,
            check_point=mock_cp,
            config=mock_cnf,
            early_stopping=mock_es,
            plotting=mock_plt,
            device=mock_device,
            epochs=10  
        )
    
    def test_new_controller(self):
        
        self.assertIsNotNone(self.ctrl)
        self.assertIsInstance(self.ctrl, Controller)
        
    
    # @patch('src.Model.CustomEfficientNet')
    def test_adaptation(self):
        
        self.ctrl.adaptation()
        
        self.assertEqual(self.ctrl.model.freeze.call_count, 1)
    
    def test_fine_tunning(self):
        
        self.ctrl.fine_tunning()
        
        self.assertEqual(self.ctrl.model.unfreeze.call_count, 1)
    
        