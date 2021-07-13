from time import time

import torch
import sklearn.datasets
import sklearn.preprocessing
import sklearn.model_selection
import numpy as np
import pandas as pd

import onlinehd

# loads simple mnist dataset
def load():
    # train data
    data=pd.read_csv('C:/onlinehd/proj_data/train_set.csv')
    x = data.drop('y',axis=1)
    x = x.astype(float)
    y = data['y']
    y = y.astype(int)
    y = np.array(y)
    # test data
    data=pd.read_csv('C:/onlinehd/proj_data/test_set.csv')
    x_test = data.drop('y',axis=1)
    x_test = x_test.astype(float)
    y_test = data['y']
    y_test = y_test.astype(int)
    y_test = np.array(y_test)

    # normalize
    scaler = sklearn.preprocessing.Normalizer().fit(x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)

    # changes data to pytorch's tensors
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    return x, x_test, y-1, y_test-1

# simple OnlineHD training
def main():
    for lr in [0.2, 0.3, 0.4, 0.5]:
        for epoch in [20,40,60]:
            for dim in [5000,7500,10000]:
                for bs in [0.25,0.5]:
                    print("Hyperparameters: lr={},epoch={},dim={},bootstrap={}".format(lr,epoch,dim,bs) )
                    print('Loading...')
                    x, x_test, y, y_test = load()
                    classes = y.unique().size(0)
                    features = x.size(1)
                    model = onlinehd.OnlineHD(classes, features,dim=dim)  # default; dim=10000
                    
                    if torch.cuda.is_available():
                        x = x.cuda()
                        y = y.cuda()
                        x_test = x_test.cuda()
                        y_test = y_test.cuda()
                        model = model.to('cuda')
                        print('Using GPU!')
                        
                    print('Training...')
                    t = time()
                    model = model.fit(x, y, bootstrap=bs, lr=lr, epochs=epoch)
                    t = time() - t
                    
                    print('Validating...')
                    yhat = model(x)
                    yhat_test = model(x_test)
                    acc = (y == yhat).float().mean()
                    acc_test = (y_test == yhat_test).float().mean()
                    print(f'{acc = :6f}')
                    print(f'{acc_test = :6f}')
                    print(f'{t = :6f}')

                    

if __name__ == '__main__':
    main()
