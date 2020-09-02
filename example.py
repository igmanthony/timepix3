'''Simple example python script showing the functionality of the timepix3.pyd
library'''

import os
import numpy as np
import timepix3 as tpx3
import matplotlib.pyplot as plt

# print(help(tpx3)) # will give data structure of the "load" function

file_path = os.path.abspath(__file__)
directory = os.path.dirname(file_path)
example_dir = os.path.join(directory, 'example_data')
for file in os.listdir(example_dir):
    if file.endswith(".tpx3"):
        data = tpx3.load(os.path.join(example_dir, file))
        labels = tpx3.cluster(data)