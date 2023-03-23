import gc
import argparse
import os
import numpy as np
import pandas as pd
import random
from sklearn import preprocessing
from helps import df_all_creator, data_creator, Input_Gen


seed = 0
random.seed(0)
np.random.seed(seed)
current_dir = os.path.dirname(os.path.abspath(__file__))
data_filedir = os.path.join(current_dir, '..','N-CMAPSS')
data_filepath = os.path.join(current_dir, '..','N-CMAPSS', 'N-CMAPSS_DS03-012.h5')




def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='sample creator')
    parser.add_argument('-w', type=int, default=50, help='window length')
    parser.add_argument('-s', type=int, default=1, help='stride of window')
    parser.add_argument('--sampling', type=int, default=10, help='sub sampling of the given data. If it is 10, then this indicates that we assumes 0.1Hz of data collection')

    args = parser.parse_args()

    sequence_length = args.w
    stride = args.s
    sampling = args.sampling



    # Load data
    '''
    W: operative conditions (Scenario descriptors)
    X_s: measured signals
    X_v: virtual sensors
    T(theta): engine health parameters
    Y: RUL [in cycles]
    A: auxiliary data
    '''

    df_all = df_all_creator(data_filepath, sampling)

    units_index = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0]



    df_data = data_creator(df_all, units_index)
    print(df_data)
    print(df_data.columns)
    print("num of inputs: ", len(df_data.columns) )

    del df_all
    gc.collect()
    df_all = pd.DataFrame()
    sample_dir_path = os.path.join(data_filedir, 'Samples_whole')
    sample_folder = os.path.isdir(sample_dir_path)
    if not sample_folder:
        os.makedirs(sample_dir_path)
        print("created folder : ", sample_dir_path)

    cols_normalize = df_data.columns.difference(['RUL', 'unit'])
    sequence_cols = df_data.columns.difference(['RUL', 'unit'])


    for unit_index in units_index:
        data_class = Input_Gen (df_data,  cols_normalize, sequence_length, sequence_cols, sample_dir_path,
                                unit_index, sampling, stride =stride)
        data_class.seq_gen()




if __name__ == '__main__':
    main()