import unittest
# from .context import src
import Model
from unittest.mock import patch
from unittest import mock




class TestCustomEfficientNet(unittest.TestCase):
    
    @patch('utils.get_EfficientNet')
    def test_new_CustomEfficientNet(self, mock_get_EfficientNet):
        mock_EfficientNet = mock.Mock()
        mock_get_EfficientNet.return_value = mock_EfficientNet
        got = Model.CustomEfficientNet()
        self.assertIsNotNone(got)
    
    
    def test_freeze(self):
        model: Model.CustomEfficientNet = Model.CustomEfficientNet()
        model.freeze()
        self.assertFalse(all(param.requires_grad for param in model.model.features.parameters()))
    
    def test_unfreeze(self):
        model = Model.CustomEfficientNet()
        model.unfreeze()
        self.assertTrue(all(param.requires_grad for param in model.parameters()))