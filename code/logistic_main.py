#########################################################################################
#Description:
# Read the traning data and perform the calculations
#
#########################################################################################
import numpy as np
from data_loader import *
(trains_image,train_images_label) =read_gz();
numpy.savez("testfile.npz", test_array_1=test_array_1, test_array_2=test_array_2)