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

newest = max(glob.iglob('logs/*/*.hdf5'), key=os.path.getctime)
dirpath0 = ('img/val3',)
dirpath1 = ('img/val4',)
dirpath2 = ('img/val5',)
dirpath3 = ('img/train2',)
dirpath4 = ('img/train3',)
dirpath5 = ('img/train4',)
dirpath6 = ('img/train5',)

dirpaths = (dirpath0, dirpath1, dirpath2, dirpath3, dirpath4, dirpath5, dirpath6,)

print("Model: " + newest.split('/')[1])



for dirpath in dirpaths:
    print("Testowane foldery: "+ str(dirpath))
    instance = predict()
    instance.load(newest)
    instance.collect_data(dirpath)
    instance.decode()
    instance.calculate_accuracy(False)
