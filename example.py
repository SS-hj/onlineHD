from time import time

import torch
import sklearn.datasets
import sklearn.preprocessing
import sklearn.model_selection
import numpy as np

import onlinehd

# loads simple mnist dataset
def load():
    # fetches data
    # sklearn에서 제공하는 minst dataset 사용
    x, y = sklearn.datasets.fetch_openml('mnist_784', return_X_y=True)
    # astype: numpy 패키지의 함수로, 데이터형 변경시 사용
    x = x.astype(float)
    y = y.astype(int)
    y = np.array(y)

    # split and normalize
    x, x_test, y, y_test = sklearn.model_selection.train_test_split(x, y)  # default 비율; train: 0.75 / test data: 0.25
    scaler = sklearn.preprocessing.Normalizer().fit(x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)

    # changes data to pytorch's tensors
    # from_numpy: numpy array인 ndarray로부터 텐서를 만듦. from_numpy로 만들어진 텐서는 해당 ndarray와 메모리를 공유하며, 어느 한쪽의 데이터를 변경 시 둘 다 변경됨
    x = torch.from_numpy(x).float() #텐서에다가 .float()를 붙이면 바로 float형으로 타입이 변경됨
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float() 
    y_test = torch.from_numpy(y_test).long()
    return x, x_test, y, y_test

# simple OnlineHD training
def main():
    print('Loading...')
    x, x_test, y, y_test = load()
    # unique: unique한 값만 ndarray로 반환
    # size: 개수 반환  (e.g., torch.Size([10]))
    # size(0)으로 하면, 해당 리스트에서 인덱스 0인 값을 빼냄 (e.g., 10)
    classes = y.unique().size(0)
    features = x.size(1) #torch.Size([행 개수, 열 개수])라 열 개수 반환
    # onlinehd 파일을 보면, default dim은 4000
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
    # bootstrap:
    model = model.fit(x, y, bootstrap=1.0, lr=0.035, epochs=20)
    t = time() - t #학습완료한 시간-시작시간

    print('Validating...')
    yhat = model(x)
    yhat_test = model(x_test)
    acc = (y == yhat).float().mean()
    acc_test = (y_test == yhat_test).float().mean()
    print(f'{acc = :6f}')
    print(f'{acc_test = :6f}')
    print(f'{t = :6f}')

# import를 안 하면  '__name__ ==  __main__' 임
# 모듈내의 아무거나 import 를 하면 '__name__==모듈이름' 이 되어, '__name__ !=  __main__' 이 됨
# 즉, import하여 해당 파일이 실행되는 경우에는 main()함수만 실행됨
if __name__ == '__main__':
    main()
