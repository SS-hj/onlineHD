from time import time

import torch
import sklearn.datasets
import sklearn.preprocessing
import sklearn.model_selection
import numpy as np

import onlinehd

import pandas as pd

# loads simple mnist dataset
def load():
    # fetches data
    # Using minst dataset provided by sklearn
    x, y = sklearn.datasets.fetch_openml('mnist_784', return_X_y=True)
    # astype: Function of the numpy package; use when changing data type
    x = x.astype(float)
    y = y.astype(int)
    y = np.array(y)

    # split and normalize
    x, x_test, y, y_test = sklearn.model_selection.train_test_split(x, y)  # default; train: 0.75 / test data: 0.25
    #scaler = sklearn.preprocessing.Normalizer().fit(x)
    #x = scaler.transform(x)
    #x_test = scaler.transform(x_test)
    x = normalizer(x)
    x_test = normalizer(x_test)

    # changes data to pytorch's tensors
    # from_numpy: Created tensor from ndarray. A tensor created with from_numpy shares memory with that ndarray
    x = torch.from_numpy(x).float() 
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float() 
    y_test = torch.from_numpy(y_test).long()
    return x, x_test, y, y_test

def normalizer(x):
    x = np.array(x)
    max_value = np.max(x)
    min_value = np.min(x)
    return (x-min_value)/(max_value-min_value)

def denormalizer(norm_x):
    norm_x = np.array(norm_x)
    max_value = np.max(norm_x)
    min_value = np.min(norm_x)
    return norm_x*(max_value-min_value) + min_value

# simple OnlineHD training
def main():
    print('Loading...')
    x, x_test, y, y_test = load()
    # unique: Return only unique values to ndarray
    # size: Return Count  (e.g., torch.Size([10]))
    classes = y.unique().size(0) # Return the value with index 0 from the list (e.g., 10)
    features = x.size(1)   #torch.Size([row num, col num]) ; Return col num
    # default dim; 4000
    model = onlinehd.OnlineHD(classes, features) #OnlineHD initialize

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        x_test = x_test.cuda()
        y_test = y_test.cuda()
        model = model.to('cuda')
        print('Using GPU!')

    print('Training...')
    t = time()
    model = model.fit(x, y, bootstrap=1.0, lr=0.035, epochs=20)
    t = time() - t
    print(f'{t = :6f}')

    print('Validating...')
    yhat = model(x)
    acc = (y == yhat).float().mean()
    print(f'{acc = :6f}')

    print('Attacking...')
    t = time()
    attacked_x = model.backprop(x)
    t = time() - t
    print(f'{t = :6f}')

    print('Attacked Validating...')
    attacked_yhat = model(attacked_x)
    acc = (y == attacked_yhat).float().mean()
    print(f'{acc = :6f}')


# if didn't use import, then '__name__ ==  __main__'
# ifelse '__name__ !=  __main__'
if __name__ == '__main__':
    main()
