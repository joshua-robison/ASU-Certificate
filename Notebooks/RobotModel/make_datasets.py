# -*- coding: utf-8 -*-
"""
==============================================================================
Neural Network for Collision Prediction
    
    Function:
        
        generate_datasets

This function loads collected data and makes training and validation sets.

"""
#-----------------------------------------------------------------------------
# Imports
import torch
from torch.utils.data import TensorDataset, DataLoader

import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#-----------------------------------------------------------------------------
# Data Preprocessing
def generate_datasets(batch_size):
    
    # load and shuffle data
    data = np.genfromtxt('saved/practice.csv', delimiter=',')
    
    # balance class sets: upsampling
    class0_idx = np.where(data[:,-1] == 0)[0]
    class1_idx = np.where(data[:,-1] == 1)[0]
    
    n_class0 = len(class0_idx)
    n_class1 = len(class1_idx)
    
    if n_class0 > n_class1:
        class1_idx = np.random.choice(class1_idx, size=n_class0, replace=True)
    elif n_class1 > n_class0:
        class0_idx = np.random.choice(class0_idx, size=n_class1, replace=True)
    
    class0_subset = data[class0_idx,:]
    class1_subset = data[class1_idx,:]
    
    assert class0_subset.shape == class1_subset.shape
    
    # remake dataset
    data = np.vstack((class0_subset, class1_subset))
    np.random.shuffle(data)
    
    # scale data
    transformer = MinMaxScaler()
    data = transformer.fit_transform(data)
    
    # save scaler
    pickle.dump(transformer, open('saved/scaler.pkl', 'wb'))
    
    # features, labels
    X, y = data[:,:-1], data[:,-1]
    
    # split into training validation and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    
    # check splitting shapes
    assert X_train.shape[0] == y_train.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    assert X_test.shape[0] == y_test.shape[0]
    
    # convert to tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()
    
    # make datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    # print(len(train_dataset), len(test_dataset))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # print(len(train_loader), len(test_loader))
    
    return train_loader, test_loader

#-----------------------------------------------------------------------------
# Main Method
def main():
    # parameters
    batch_size = 16
    
    # generate data
    train_loader, test_loader = generate_datasets(batch_size)

#-----------------------------------------------------------------------------
if __name__ == '__main__':
    main()

