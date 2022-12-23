from calendar import c
import unittest
from unittest.mock import Mock, patch

from src.Plotting import Plotting
import matplotlib.pyplot as plt

class TestPlotting(unittest.TestCase):
    
    def setUp(self) -> None:
        self.plt = Plotting(10)
        self.plt.plot_tick = list(range(10))
    
    
    def test_new_Plotting(self):
        self.assertIsNotNone(self.plt)
        self.assertIsInstance(self.plt, Plotting)
     
    # Plotting@_plot_func
   
    # private function i have no idea to test it through public function 
    def test_plot_func_plot(self):
        got = self.plt._plot_func(scale='linear')
        
        self.assertIsNotNone(got)
        self.assertEqual(got, plt.plot)
    # private function         
    def test_plot_func_semilogy(self):
        got = self.plt._plot_func(scale='semilogy')
        
        self.assertIsNotNone(got)
        self.assertEqual(got, plt.semilogy)
        
    # TODO create unit test for Plotting@_plot_func (Error test case)
    
    
    def test_reset_tick(self):
        # assert plot tick is not empty using truthy value before function to tested is called 
        self.assertTrue(self.plt.plot_tick)
        
        self.plt.reset_tick()
        
        # check if plot tick is empty using truthy value
        self.assertFalse(self.plt.plot_tick)
    
    def test_add_epoch_to_tick(self):
        len_before = len(self.plt.plot_tick)
        
        self.plt.add_epoch_to_tick(10)
        
        self.assertEqual(len(self.plt.plot_tick), len_before + 1)
    
    # TODO create unit test for others functions which produce side effect
    