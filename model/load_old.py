import numpy
import glob

from analyse.analyse_old import Struct

struc = Struct(window=numpy.loadtxt(f'TWO_DIR/EXAMPLE_25/Window_Parameter.txt', comments='%'),
               record_files=sorted(glob.glob(f'TWO_DIR/EXAMPLE_25/Struc_fct_records/*_part_1.txt')),
               gamma=numpy.loadtxt(f'TWO_DIR/EXAMPLE_25/systemparameter.txt', comments='%')[5])
