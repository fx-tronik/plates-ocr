import os
import glob
from detekcja import predict
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-g,', '--gpu', help='Which gpu to use', type=str, default="1")

args = vars(parser.parse_args())
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]=args['gpu']

newest = 'best_weights.hdf5'
dirpath = ('img/test',)

instance = predict()

instance.load(newest)
instance.collect_data(dirpath)
instance.decode()
instance.display_results()
