import os
import glob
from detekcja import predict

newest = 'best_weights.hdf5'
dirpath = ('img/test',)

instance = predict()

instance.load(newest)
instance.collect_data(dirpath)
instance.decode()
instance.display_results()
