from importlib.resources import path
import unittest
from unittest.mock import patch, MagicMock
from src.EarlyStopping import EarlyStopping

class TestEarlyStopping(unittest.TestCase):
    
    @patch('src.Config.Config')
    def setUp(self, mock_config) -> None:
        self.ea = EarlyStopping(cnf=mock_config, max_counter=10, load_best_when_stop=True, patience=1)
    
    def test_new_EarlyStopping(self):
        
        self.assertIsNotNone(self.ea)
        self.assertIsInstance(self.ea, EarlyStopping)
    
    def test_set_patience(self):
        self.ea.set_patience(10_000)
        
        self.assertEqual(self.ea.early_stop_patience, 10_000)
    
    def test_reset_early_stop(self):
        
        self.ea.reset_early_stop()
        
        self.assertEqual(self.ea._early_stop_counter, 0)
    
    # private function
    @patch('torch.save')
    @patch('src.CheckPoint.CheckPoint')
    def test_save_weights(self, mock_cp, mock_torch_save):
        
        self.ea._save_weight(ckpt=mock_cp)
        
        self.assertEqual(mock_torch_save.call_count, 1)
    
    @patch('src.CheckPoint.CheckPoint')
    @patch('torch.nn.Module')
    @patch('torch.save')
    def test_is_stop_False(self, _, mock_nn_Module, mock_cp):
        
        mock_cp.best_acc = 0.77
        mock_cp.test_acc = [0.77, 0.82]
        
        got = self.ea.is_stop(model=mock_nn_Module, ckpt=mock_cp, counter=1, device='cpu', mode='test_acc')
        
        self.assertFalse(got)
        
    
    @patch('src.CheckPoint.CheckPoint')
    @patch('torch.nn.Module')
    @patch('torch.save')
    @patch('torch.load')
    @patch('builtins.print')
    def test_is_stop_True(self, _3, _2, _, mock_nn_Module, mock_cp):
        
        mock_cp.best_acc = 0.77
        mock_cp.test_acc = [0.77, 0.66]
        self.ea._early_stop_counter = 4
        self.ea.max_counter = 5
        
        got = self.ea.is_stop(model=mock_nn_Module, ckpt=mock_cp, counter=10, device='cpu', mode='test_acc')
        
        self.assertTrue(got)
        
    # @patch('torch.nn.Module')
    # @patch('src.CheckPoint.CheckPoint')
    # def test_is_stop_error(self, mock_cp, mock_nn_Module):
        
        
    #     # got = self.ea.is_stop(model=mock_nn_Module, ckpt=mock_cp, counter=10, device='cpu', mode='non-supported-mode')
        
    #     self.assertRaisesRegex(Exception, "can't support mode:.*", self.ea.is_stop(mock_nn_Module, mock_cp, 1, 'cpu', 'not supported mode'))
    
    
    