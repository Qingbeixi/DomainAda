import os
import h5py
import time
import numpy as np
import pandas as pd
from random import shuffle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def df_all_creator(data_filepath, sampling):
    """
     """
    # Time tracking, Operation time (min):  0.003
    t = time.process_time()


    with h5py.File(data_filepath, 'r') as hdf:
        # Development(training) set
        W_dev = np.array(hdf.get('W_dev'))  # W
        X_s_dev = np.array(hdf.get('X_s_dev'))  # X_s
        X_v_dev = np.array(hdf.get('X_v_dev'))  # X_v
        T_dev = np.array(hdf.get('T_dev'))  # T
        Y_dev = np.array(hdf.get('Y_dev'))  # RUL
        A_dev = np.array(hdf.get('A_dev'))  # Auxiliary

        # Test set
        W_test = np.array(hdf.get('W_test'))  # W
        X_s_test = np.array(hdf.get('X_s_test'))  # X_s
        X_v_test = np.array(hdf.get('X_v_test'))  # X_v
        T_test = np.array(hdf.get('T_test'))  # T
        Y_test = np.array(hdf.get('Y_test'))  # RUL
        A_test = np.array(hdf.get('A_test'))  # Auxiliary

        # Varnams
        W_var = np.array(hdf.get('W_var'))
        X_s_var = np.array(hdf.get('X_s_var'))
        X_v_var = np.array(hdf.get('X_v_var'))
        T_var = np.array(hdf.get('T_var'))
        A_var = np.array(hdf.get('A_var'))

        # from np.array to list dtype U4/U5
        W_var = list(np.array(W_var, dtype='U20'))
        X_s_var = list(np.array(X_s_var, dtype='U20'))
        X_v_var = list(np.array(X_v_var, dtype='U20'))
        T_var = list(np.array(T_var, dtype='U20'))
        A_var = list(np.array(A_var, dtype='U20'))


    W = np.concatenate((W_dev, W_test), axis=0)
    X_s = np.concatenate((X_s_dev, X_s_test), axis=0)
    X_v = np.concatenate((X_v_dev, X_v_test), axis=0)
    T = np.concatenate((T_dev, T_test), axis=0)
    Y = np.concatenate((Y_dev, Y_test), axis=0)
    A = np.concatenate((A_dev, A_test), axis=0)

    print('')
    print("Operation time (min): ", (time.process_time() - t) / 60)
    print("number of training samples(timestamps): ", Y_dev.shape[0])
    print("number of test samples(timestamps): ", Y_test.shape[0])
    print('')
    print("W shape: " + str(W.shape))
    print("X_s shape: " + str(X_s.shape))
    print("X_v shape: " + str(X_v.shape))
    print("T shape: " + str(T.shape))
    print("Y shape: " + str(Y.shape))
    print("A shape: " + str(A.shape))

    '''
    Illusration of Multivariate time-series of condition monitoring sensors readings for Unit5 (fifth engine)
    W: operative conditions (Scenario descriptors) - ['alt', 'Mach', 'TRA', 'T2']
    X_s: measured signals - ['T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf']
    X_v: virtual sensors - ['T40', 'P30', 'P45', 'W21', 'W22', 'W25', 'W31', 'W32', 'W48', 'W50', 'SmFan', 'SmLPC', 'SmHPC', 'phi']
    T(theta): engine health parameters - ['fan_eff_mod', 'fan_flow_mod', 'LPC_eff_mod', 'LPC_flow_mod', 'HPC_eff_mod', 'HPC_flow_mod', 'HPT_eff_mod', 'HPT_flow_mod', 'LPT_eff_mod', 'LPT_flow_mod']
    Y: RUL [in cycles]
    A: auxiliary data - ['unit', 'cycle', 'Fc', 'hs']
    '''

    df_W = pd.DataFrame(data=W, columns=W_var)
    df_Xs = pd.DataFrame(data=X_s, columns=X_s_var)
    df_Xv = pd.DataFrame(data=X_v[:,0:2], columns=['T40', 'P30'])
    # df_T = pd.DataFrame(data=T, columns=T_var)
    df_Y = pd.DataFrame(data=Y, columns=['RUL'])
    df_A = pd.DataFrame(data=A, columns=A_var).drop(columns=['cycle', 'Fc', 'hs'])



    # Merge all the dataframes
    df_all = pd.concat([df_W, df_Xs, df_Xv, df_Y, df_A], axis=1)

    print ("df_all", df_all)    # df_all = pd.concat([df_W, df_Xs, df_Xv, df_Y, df_A], axis=1).drop(columns=[ 'P45', 'W21', 'W22', 'W25', 'W31', 'W32', 'W48', 'W50', 'SmFan', 'SmLPC', 'SmHPC', 'phi', 'Fc', 'hs'])

    print ("df_all.shape", df_all.shape)
    # del [[df_W, df_Xs, df_Xv, df_Y, df_A]]
    # gc.collect()
    # df_W = pd.DataFrame()
    # df_Xs = pd.DataFrame()
    # df_Xv = pd.DataFrame()
    # df_Y = pd.DataFrame()
    # df_A = pd.DataFrame()

    df_all_smp = df_all[::sampling]
    print ("df_all_sub", df_all_smp)    # df_all = pd.concat([df_W, df_Xs, df_Xv, df_Y, df_A], axis=1).drop(columns=[ 'P45', 'W21', 'W22', 'W25', 'W31', 'W32', 'W48', 'W50', 'SmFan', 'SmLPC', 'SmHPC', 'phi', 'Fc', 'hs'])

    print ("df_all_sub.shape", df_all_smp.shape)


    return df_all_smp



def data_creator(df_all, units_index):
    df_lst= []
    for idx in units_index:
        df_temp = df_all[df_all['unit'] == np.float64(idx)]
        df_lst.append(df_temp)
    data = pd.concat(df_lst)
    data = data.reset_index(drop=True)

    return data


def gen_sequence(id_df, seq_length, seq_cols):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    # for one id I put all the rows in a single matrix
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    # Iterate over two lists in parallel.
    # For example id1 have 192 rows and sequence_length is equal to 50
    # so zip iterate over two following list of numbers (0,142),(50,192)
    # 0 50 -> from row 0 to row 50
    # 1 51 -> from row 1 to row 51
    # 2 52 -> from row 2 to row 52
    # ...
    # 142 192 -> from row 142 to 192
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]


def gen_labels(id_df, seq_length, label):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    # For one id I put all the labels in a single matrix.
    # For example:
    # [[1]
    # [4]
    # [1]
    # [5]
    # [9]
    # ...
    # [200]]
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    # I have to remove the first seq_length labels
    # because for one id the first sequence of seq_length size have as target
    # the last label (the previus ones are discarded).
    # All the next id's sequences will have associated step by step one label as target.
    return data_matrix[seq_length:num_elements, :]

def time_window_slicing (input_array, sequence_length, sequence_cols):
    # generate labels
    label_gen = [gen_labels(input_array[input_array['unit'] == id], sequence_length, ['RUL'])
                 for id in input_array['unit'].unique()]
    label_array = np.concatenate(label_gen).astype(np.float32)
    # label_array = np.concatenate(label_gen)

    # transform each id of the train dataset in a sequence
    seq_gen = (list(gen_sequence(input_array[input_array['unit'] == id], sequence_length, sequence_cols))
               for id in input_array['unit'].unique())
    sample_array = np.concatenate(list(seq_gen)).astype(np.float32)
    # sample_array = np.concatenate(list(seq_gen))

    print("sample_array")
    return sample_array, label_array


def time_window_slicing_label_save (input_array, sequence_length, stride, index, sample_dir_path, sequence_cols = 'RUL'):
    '''
    ref
        for i in range(0, input_temp.shape[0] - sequence_length):
        window = input_temp[i*stride:i*stride + sequence_length, :]  # each individual window
        window_lst.append(window)
        # print (window.shape)
    '''
    # generate labels
    window_lst = []  # a python list to hold the windows

    input_temp = input_array[input_array['unit'] == index][sequence_cols].values
    num_samples = int((input_temp.shape[0] - sequence_length)/stride) + 1
    for i in range(num_samples):
        window = input_temp[i*stride:i*stride + sequence_length]  # each individual window
        window_lst.append(window)

    label_array = np.asarray(window_lst).astype(np.float32)

    # np.save(os.path.join(sample_dir_path, 'Unit%s_rul_win%s_str%s' %(str(int(index)), sequence_length, stride)),
    #         label_array)  # save the file as "outfile_name.npy"

    return label_array[:,-1]



def time_window_slicing_sample_save (input_array, sequence_length, stride, index, sample_dir_path, sequence_cols):
    '''
    '''
    # generate labels
    window_lst = []  # a python list to hold the windows

    input_temp = input_array[input_array['unit'] == index][sequence_cols].values
    print ("Unit%s input array shape: " %index, input_temp.shape)
    num_samples = int((input_temp.shape[0] - sequence_length)/stride) + 1
    for i in range(num_samples):
        window = input_temp[i*stride:i*stride + sequence_length,:]  # each individual window
        window_lst.append(window)

    sample_array = np.dstack(window_lst).astype(np.float32)
    # sample_array = np.dstack(window_lst)
    print ("sample_array.shape", sample_array.shape)

    # np.save(os.path.join(sample_dir_path, 'Unit%s_samples_win%s_str%s' %(str(int(index)), sequence_length, stride)),
    #         sample_array)  # save the file as "outfile_name.npy"


    return sample_array



class Input_Gen(object):
    '''
    class for data preparation (sequence generator)
    '''

    def __init__(self, df, cols_normalize, sequence_length, sequence_cols, sample_dir_path,
                 unit_index, sampling, stride):
        '''
        '''
        # self.__logger = logging.getLogger('data preparation for using it as the network input')
        print("the number of input signals: ", len(cols_normalize))
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        
        norm_df = pd.DataFrame(min_max_scaler.fit_transform(df[cols_normalize].values),
                               columns=cols_normalize,
                               index=df.index)
        join_df = df[df.columns.difference(cols_normalize)].join(norm_df)
        df = join_df.reindex(columns=df.columns)

        self.df = df
        print (self.df)

        self.cols_normalize = cols_normalize
        self.sequence_length = sequence_length
        self.sequence_cols = sequence_cols
        self.sample_dir_path = sample_dir_path
        self.unit_index = np.float64(unit_index)
        self.sampling = sampling
        self.stride = stride


    def seq_gen(self):
        '''
        concatenate vectors for NNs
        :param :
        :param :
        :return:
        '''

        label_array = time_window_slicing_label_save(self.df, self.sequence_length,
                                        self.stride, self.unit_index, self.sample_dir_path, sequence_cols='RUL')
        sample_array = time_window_slicing_sample_save(self.df, self.sequence_length,
                                        self.stride, self.unit_index, self.sample_dir_path, sequence_cols=self.cols_normalize)


        # sample_split_lst = np.array_split(sample_array, 3, axis=2)
        # print (sample_split_lst[0].shape)
        # print(sample_split_lst[1].shape)
        # print(sample_split_lst[2].shape)

        # label_split_lst = np.array_split(label_array, 3, axis=0)
        # print (label_split_lst[0].shape)
        # print(label_split_lst[1].shape)
        # print(label_split_lst[2].shape)

        print("sample_array.shape", sample_array.shape)
        print("label_array.shape", label_array.shape)



        np.savez_compressed(os.path.join(self.sample_dir_path, 'Unit%s_win%s_str%s_smp%s' %(str(int(self.unit_index)), self.sequence_length, self.stride, self.sampling)),
                                         sample=sample_array, label=label_array)
        print ("unit saved")

        return


def load_part_array (sample_dir_path, unit_num, win_len, stride, part_num):
    filename =  'Unit%s_win%s_str%s_part%s.npz' %(str(int(unit_num)), win_len, stride, part_num)
    filepath =  os.path.join(sample_dir_path, filename)
    loaded = np.load(filepath)
    return loaded['sample'], loaded['label']

def load_part_array_merge (sample_dir_path, unit_num, win_len, win_stride, partition):
    sample_array_lst = []
    label_array_lst = []
    print ("Unit: ", unit_num)
    for part in range(partition):
      print ("Part.", part+1)
      sample_array, label_array = load_part_array (sample_dir_path, unit_num, win_len, win_stride, part+1)
      sample_array_lst.append(sample_array)
      label_array_lst.append(label_array)
    sample_array = np.dstack(sample_array_lst)
    label_array = np.concatenate(label_array_lst)
    sample_array = sample_array.transpose(2, 0, 1)
    print ("sample_array.shape", sample_array.shape)
    print ("label_array.shape", label_array.shape)
    return sample_array, label_array


def load_array (sample_dir_path, unit_num, win_len, stride, sampling):
    filename =  'Unit%s_win%s_str%s_smp%s.npz' %(str(int(unit_num)), win_len, stride, sampling)
    filepath =  os.path.join(sample_dir_path, filename)
    loaded = np.load(filepath)

    return loaded['sample'].transpose(2, 0, 1), loaded['label']


def shuffle_array(sample_array, label_array):
    ind_list = list(range(len(sample_array)))
    print("ind_list befor: ", ind_list[:10])
    print("ind_list befor: ", ind_list[-10:])
    ind_list = shuffle(ind_list)
    print("ind_list after: ", ind_list[:10])
    print("ind_list after: ", ind_list[-10:])
    print("Shuffeling in progress")
    shuffle_sample = sample_array[ind_list, :, :]
    shuffle_label = label_array[ind_list,]
    return shuffle_sample, shuffle_label



def scheduler(epoch, lr):
    if epoch == 30:
        print("lr decay by 10")
        return lr * 0.1
    elif epoch == 70:
        print("lr decay by 10")
        return lr * 0.1
    else:
        return lr



def release_list(a):
   del a[:]
   del a


def load_data(sample_dir, args):
    """
    Load and preprocess data for training and testing the models.
    
    Args:
    - data_dir: Directory where the data file is located
    - data_filename: Name of the data file
    - sample_dir: Directory where the preprocessed data is saved
    
    Returns:
    - sample_dict: A dictionary containing preprocessed data for each unit group
    - label_dict: A dictionary containing preprocessed labels for each unit group
    """
    sample_dict = {}
    label_dict = {}
    units_all = [[1, 5, 9, 12, 14], [2, 3, 4, 7, 15], [6, 8, 10, 11, 13]]
    EOF = [72, 73, 67, 60, 93, 63, 80, 71, 84, 66, 59, 93, 77, 76, 67]

    for i, units in enumerate(units_all):
        sample_list = []
        sample_label_list = []
        for id, index in enumerate(units):
            sample_array, label_array = load_array(sample_dir, index, args.w, args.s, args.sampling)
            sample_array = sample_array[::args.sub]
            label_array = label_array[::args.sub]
            sample_list.append(sample_array)
            sample_label_list.append(label_array / EOF[index - 1])

        X_sample = np.concatenate(sample_list)
        y_sample_label = np.concatenate(sample_label_list).reshape(-1, 1)
        sample_dict[i] = X_sample
        label_dict[i] = y_sample_label

        # release memory
        release_list(sample_list)
        release_list(sample_label_list)

    return sample_dict, label_dict

def load_train_test_data(sample_dir, args):
    """
    Load and preprocess data for training and testing the models.
    
    Args:
    - data_dir: Directory where the data file is located
    - data_filename: Name of the data file
    - sample_dir: Directory where the preprocessed data is saved
    
    Returns:
    - sample_dict: A dictionary containing preprocessed data for each unit group
    - label_dict: A dictionary containing preprocessed labels for each unit group
    - test_dict:  A dictionary containing preprocessed labels for each unit 
    """
    sample_dict = {}
    label_dict = {}
    test_dict = {}
    test_label_dict = {}
    units_all = [[1, 5, 9, 12, 14], [2, 3, 4, 7, 15], [6, 8, 10, 11, 13]]
    EOF = [72, 73, 67, 60, 93, 63, 80, 71, 84, 66, 59, 93, 77, 76, 67]

    for i, units in enumerate(units_all):
        sample_list = []
        sample_label_list = []
       
        for id, index in enumerate(units):
            sample_array, label_array = load_array(sample_dir, index, args.w, args.s, args.sampling)
            sample_array = sample_array[::args.sub]
            label_array = label_array[::args.sub]
            
            
            x_train,x_test,y_train,y_test = train_test_split(sample_array,label_array,test_size=0.2,random_state=42,shuffle=True)
            sample_list.append(x_train)
            sample_label_list.append(y_train / EOF[index - 1])
            test_dict[index] =  x_test
            test_label_dict[index] = y_test / EOF[index - 1]

        X_sample = np.concatenate(sample_list)
        y_sample_label = np.concatenate(sample_label_list).reshape(-1, 1)
        sample_dict[i] = X_sample
        label_dict[i] = y_sample_label

        # release memory
        release_list(sample_list)
        release_list(sample_label_list)

    return sample_dict, label_dict, test_dict, test_label_dict