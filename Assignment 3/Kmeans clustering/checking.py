import numpy as np
import csv
import pandas as pd
from tqdm import tqdm
import math
import random

distance = np.load("clusters_k_means.dat", allow_pickle=True)
print(np.sort((np.array(distance[0]))))
print(np.sort((np.array(distance[1]))))
print(np.sort((np.array(distance[2]))))
print(np.sort((np.array(distance[3]))))
print(np.sort((np.array(distance[4]))))
print(np.sort((np.array(distance[5]))))
print(np.sort((np.array(distance[6]))))
print(np.sort((np.array(distance[7]))))