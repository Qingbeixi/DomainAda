# units: [2,5,10,11,14,15,16,18,20]
# small [14] 1-3h
# medium [15] 3-5h
# long [2,5,10,11,16,18,20] 5-7h

"We propose to train on one set and test on another"
"Document: sample numbers, losses(rmse) for train and test after each epoch"
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dir_path, '..'))


from models.repetition import *
import torch
from models.VAE import *
from models.ExtractorRegressor import *
import numpy as np
import torch.optim as optim
from src.myDataset import *
from torchvision import transforms, datasets
import argparse
import torch.optim
from src.helps import *
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
from models.AE import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
"""get the path to files"""
current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.dirname(current_dir)
data_filedir = os.path.join(current_dir, 'N-CMAPSS')
data_filepath = os.path.join(current_dir, 'N-CMAPSS', 'N-CMAPSS_DS03-012.h5')
sample_dir_path = os.path.join(data_filedir, 'Samples_whole')
pic_dir = os.path.join(current_dir, 'Figures')

def main():
    """add parameter"""
    parser = argparse.ArgumentParser(description='sample creator')
    parser.add_argument('-model', type=str, default='ExtractorRegressor', help='model type to choose')
    parser.add_argument('-task', type=str, default='s to m', help='condition of domain adaptation')
    parser.add_argument('--sampling', type=int, default=10, help='sub sampling of the given data. If it is 10, then this indicates that we assumes 0.1Hz of data collection')
    parser.add_argument('-w', type=int, default=50, help='sequence length') # required=True
    parser.add_argument('-s', type=int, default=1, help='stride of filter')
    parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-bs', type=int, default=256, help='batch size')
    parser.add_argument('-sub', type=int, default=1, help='subsampling stride')
    parser.add_argument('-ep', type=int, default=30, help='max epoch')
    parser.add_argument('-load', type=int, default=True, help='whether load previous model')
    parser.add_argument('-freeze', type=bool, default=False, help='whether freeze encoder')
    args = parser.parse_args()

    """define each group type""" # 
    units_small = [1,5,9,12,14] # 
    units_medium = [2,3,4,7,15] # 
    units_long = [6,8,10,11,13] # 
    EOF = [72,73,67,60,93,63,80,71,84,66,59,93,77,76,67]
    domainType = args.task
    win_len = args.w
    win_stride = args.s
    lr = args.lr
    ep = args.ep
    bs = args.bs
    sub = args.sub
    sampling = args.sampling
    freeze = args.freeze
    
    units_all = [units_small,units_medium,units_long] 
    sample_dict,label_dict,test_dict, test_label_dict =  load_train_test_data(sample_dir_path, args)

    train_str = domainType[0]
    str_map = {"s":0,"m":1,"l":2}
    X_train = sample_dict[str_map[train_str]]
    y_train = label_dict[str_map[train_str]]
    X_test = []
    y_test = []
    for unit in units_all[str_map[train_str]]:
        X_test.append(test_dict[unit])
        y_test.append(test_label_dict[unit])
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test).reshape(-1, 1)


    train_dataset = TurbineDataset(X_train,y_train)
    validate_dataset = TurbineDataset(X_test,y_test)

    """prepare the model for training"""
    model = AE()
    model.load_state_dict(torch.load("documents/" + train_str + "ae_model_params.pth"))
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    optimizer = optim.Adam(model.parameters(), lr=lr, betas = (0.9,0.999), eps=1e-07, amsgrad=True)
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("We use the device",device)
    # Move the model to the GPU
    model.to(device)
    # Train the model
    num_epochs = ep
    criterion = nn.MSELoss()

    """Train the model"""
    print("We apply the domain adaptation for",domainType)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader =  DataLoader(validate_dataset, batch_size=bs, shuffle=True)
    train_document = [] # document the loss, and save to file
    val_document = []

    # Create a regressor that takes the output of the encoder as input
    regressor = AE_Regressor()
    regressor.to(device)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            encoded_inputs = model.encode(x_batch)
            # z = model.reparameterize(encoded_inputs,logvar)    
            y_pred = regressor(encoded_inputs)
            loss = torch.sqrt(criterion(y_pred, y_batch))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)

        with torch.no_grad():
            test_loss = 0
            for (x_batch, y_batch) in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                encoded_inputs = model.encode(x_batch)
                y_pred = regressor(encoded_inputs)
                val_loss = torch.sqrt(criterion(y_pred, y_batch))
                test_loss += val_loss.item() * x_batch.size(0)
        
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_val_loss = test_loss / len(validate_dataset)
        train_document.append(epoch_train_loss)
        val_document.append(epoch_val_loss)
        # print(f"Epoch {epoch+1}/{num_epochs}, Train_RMSELoss: {epoch_train_loss:.4f}, Test_RMSELoss:{epoch_val_loss:.4f}")
    
    # test results
    print("RMSE for train")
    save_dict_loss = {}
    output_lst = []
    truth_lst = []
    with torch.no_grad():
        for index in units_all[str_map[train_str]]: # units for test
                # print ("train idx: ", index)
        
                sample_array = test_dict[index]
                label_array = test_label_dict[index] * EOF[index-1] # 0-1 =>
                
                sample_tensor = torch.tensor(sample_array, dtype=torch.float32)
                sample_tensor = sample_tensor.to(device)
                encoded_inputs  = model.encode(sample_tensor)
                y_pred_test = regressor(encoded_inputs)
                y_pred_test = y_pred_test * EOF[index-1]
                rms_temp = np.sqrt(mean_squared_error(y_pred_test.cpu(), label_array))
                # print("the rms for train index {} is {}".format(index,rms_temp))
                
                # document testing loss results
                name = "Test Loss for unit " + str(index) 
                save_dict_loss[name] = float(rms_temp)
                output_lst.append(y_pred_test.cpu())
                truth_lst.append(label_array)
        output_array = np.concatenate(output_lst)[:, 0]
        trytg_array = np.concatenate(truth_lst)
        rms = np.sqrt(mean_squared_error(output_array, trytg_array))
        s_score = output_array - trytg_array
        def score(x):
            return 0.1*x if x>0 else 1/13*x
        s_score = np.mean(np.exp(np.abs(np.array(list(map(score, s_score))))))
        rms = round(rms, 2)
        print("ep",ep)
        print("lr",lr)
        print("source",domainType)
        print("rms for train",rms)   
        print("s score for train",s_score) 
                # print(output_lst[0].shape)
        

if __name__ == '__main__':
    main()