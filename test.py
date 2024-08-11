from search import *
from data_processing import *
import itertools
import numpy as np
import pprint as pp
from time import sleep

# test functions individually
# run stuff to see if it actually works as intended

df = import_data('data/small-test-dataset.txt')
df2 = import_data('data/large-test-dataset.txt')
df3 = import_data('data/CS170_Spring_2023_Small_data__28.txt')
df4 = import_data('data/CS170_Spring_2023_Large_data__28.txt')

datasets = [df, df2, df3, df4]

for d in datasets:
    print(d.shape)
    print(d[0].value_counts())
