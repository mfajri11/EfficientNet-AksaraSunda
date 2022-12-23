import unittest
from unittest.mock import patch
from src.CheckPoint import CheckPoint

class TestCheckPoint(unittest.TestCase):
    
    def setUp(self) -> None:
        self.cp = CheckPoint()
        
    
    
    def test_new_CheckPoint(self):
        
        self.assertIsNotNone(self.cp)
        self.assertIsInstance(self.cp, CheckPoint)
    
    @patch('src.Config.Config')
    @patch('torch.save')
    def test_save_weights(self, _, mock_config):
        self.cp.save_weights(cnf=mock_config, epoch=1)
        
    
    def test_log(self):
        
        before = (
            len(self.cp.train_acc),
            len(self.cp.test_acc),
            len(self.cp.train_cost),
            len(self.cp.test_cost))
        
        self.cp.log(train_acc=1, train_cost=0, test_acc=1, test_cost=0)
        
        after = (
            len(self.cp.train_acc),
            len(self.cp.test_acc),
            len(self.cp.train_cost),
            len(self.cp.test_cost))
        
        val = sum(before)
        got = sum(after)
        self.assertNotEqual(got, val)
    
    
    def test_reset(self):
        
        self.cp.log(train_acc=1, train_cost=0, test_acc=1, test_cost=0)
        
        self.cp.reset()
        got = sum((
            len(self.cp.train_acc),
            len(self.cp.test_acc),
            len(self.cp.train_cost),
            len(self.cp.test_cost)))
        self.assertEqual(got, 0)
    

        