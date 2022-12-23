import unittest
from unittest.mock import patch, mock_open, Mock
from src import Config as cfg
from src.Config import Config



class TestConfig(unittest.TestCase):
    
    _CONFIG_ARGS = {
        "base_output_dir" : "output_test",
        "model_output_dir": "output_test/model",
        "base_data_dir": "data_test",
        "dataset_name": "test_datasets_name",
        "optimizer_name": "OPT",
        "learning_rate": 1e-5,
        "epochs": 30,
        "drop_out":0.3,
        "batch_size": 64,
        "es_epochs": 100,
        "patience":5,
        "plot_every":32,
        "ratio_split": "60/40",
        "weight_decay": 1e-4,
        "load_best": True,
        "num_of_class":3
    }
    
    _GEN_CONFIG_ARGS = {
        'lrs': [1e-3, 1e-4, 1e-5],
        'eps': [30, 50],
        'opts': ['Adam', 'NAdam'],
        'is_include_es': True,
        'target': './target',
        'ratios': ['60/40', '70/30', '80/20']
    }
    
    def count_json_dumps_called(self):
        # try to count how many json.dumps is called 
        # based on the foor loop structure
        res = 1
        d = self._GEN_CONFIG_ARGS
        res *= len(d['ratios'])
        res *= len(d['opts'])
        res *= len(d['eps'])
        is_include_es = d['is_include_es']
        if is_include_es:
            res = (res) + ( res / len(d['eps']))
        res *= len(d['lrs'])
        return res
        
    
    def test_new_Config(self):
        
        got = Config(**self._CONFIG_ARGS)
        
        self.assertIsInstance(got, Config)
    
    def test_get_dict(self):
        
        cnf = Config(**self._CONFIG_ARGS)
        got = cnf.get_dict()
        
        # assert is  not empty dict
        self.assertIsInstance(got, dict)
        self.assertTrue(bool(got))
    
    # test function module 
    # create mock with patch decoretator
    @patch('builtins.open', mock_open(read_data='test'))
    @patch('pathlib.Path')
    @patch('utils.convert_str_to_path')
    @patch('json.load')
    def test_load_config(self, mock_json_load, mock_convert_str_to_path, mock_path):
        
        mock_convert_str_to_path.return_value = mock_path
        mock_path.is_file = True
        mock_json_load.return_value = self._CONFIG_ARGS
        
        got = cfg.load_config('./test.json')
        
        self.assertIsNotNone(got)
        self.assertIsInstance(got, Config)
    
    
    @patch('builtins.open', mock_open(read_data='test'))
    @patch('src.Config.Config') 
    @patch('src.Config.load_config')
    @patch('builtins.print')
    @patch('json.dumps')
    def test_generate_all_possible_config(self, mock_json_dumps,_, mock_load_config, mock_config):
        
        mock_load_config.return_value = mock_config
        mock_config.get_dict.return_value = self._CONFIG_ARGS
        call_count = self.count_json_dumps_called()
        
        cfg.generate_all_possible_config(**self._GEN_CONFIG_ARGS)
        
        self.assertEqual(mock_json_dumps.call_count, call_count)
        
        