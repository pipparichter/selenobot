from selenobot.tools import Kofamscan
from selenobot.utils import output_path
import pandas as pd 
import multiprocessing
from multiprocessing import Pool, Value, Lock
import numpy as np 
import argparse 
import os 

# TODO: Should I use a queue?
# TODO: Should I use map or map_async? map_async is non-blocking, while map is blocking.

class Counter():

    def __init__(self, total:int):

        self.total = Value('1', total)
        self.count = Value('1', 0)
        self.lock = Lock() 

    def increment(self):
        with self.lock:
            self.count.value += 1

    def value(self):
        with self.lock:
            return self.counter.value



def annotate(input_path:str, output_path:str, counter:Counter=None):
    
    kofamscan = Kofamscan()
    kofamscan.run(input_path, output_path)






if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default=None)
    parser.add_argument('--input-dir', type=str, default=None)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--n-processes', type=int, default=os.cpu_count())
    args = parser.parse_args()

    if (args.input_dir is not None):
        input_file_names = os.listdir(args.input_dir) # This is just the file names. 
        output_file_names = [default_output_path(file_name, op='ko', ext='tsv') for file_name in input_file_names]

        output_dir = args.input_dir if (args.output_dir is None) else args.output_dir
        output_paths = [os.path.join(output_dir, file_name) for file_name in output_file_names]
        input_paths = [os.path.join(args.input_dir, file_name) for file_name in input_file_names]

        pool = Pool(processes=args.n_processes)
        pool.map()

        pool.close()

    else:
        output_path = args.output_path if (args.output_path is not None) else default_output_path(args.input_path, op='ko', ext='tsv')
        annotate(args.input_path, output_path)

    pass 