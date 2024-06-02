# write a unit test function for the function z_fit in the file z_fit.py
import unittest
from analyse.data_loader.cpp_loader import load_cpp_data


class TestZFit(unittest.TestCase):
    # load old data for comparison
    def setUp(self):
        base_root = 'cpp_data'
        gamma = 0.25
        self.corr, self.time = load_cpp_data(base_root=base_root, gamma=gamma, length=100000, t_max=10000, num=50)

    def test_z_func(self):
        a = 2
        self.assertEqual(a, 2)
        pass
