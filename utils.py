import numpy as np


def predict(x, w):
    return np.matmul(x, w)

def predict_with_bias(x, w, b):
    return np.matmul(x, w) + b

def rss(y, y_hat):    
    return np.square(y_hat - y).sum()

def d_rss_w(y, y_hat, x):    
    return np.sum(np.matmul(x.T, y_hat - y), axis=0)

def d_rss_w_0(y, y_hat):    
    return np.sum(y_hat - y)

def synthetic_linear(data_size = 100, training_size = 60):
    x_0 = np.ones(data_size)
    x_1 = np.random.randint(-30, 30, data_size)
    eps = np.random.normal(0, 10, data_size)

    w_0 = -3.5
    w_1 = 2

    y = w_0 * x_0 + w_1 * x_1 + eps

    x_train = np.c_[x_0[0:training_size], x_1[0:training_size]]
    y_train = y[0:training_size]

    x_test = np.c_[x_0[training_size:data_size], x_1[training_size:data_size]]
    y_test = y[training_size:data_size]

    return x_train, y_train, x_test, y_test


def ordinary_least_squares(x_train, y_train):
    return np.matmul(np.linalg.inv(np.matmul(x_train.T, x_train)), np.matmul(x_train.T, y_train))

def squared_error(y_hat, y):
    return np.sum(np.square(y_hat - y), axis=0)

def train_gd_linear(x, y, num_steps=1000, lr=0.01):
    _w = np.random.random(x.shape[1])   
    _b = np.random.random(x.shape[1])

    _w_trail_path = [_w]
    _b_trail_path = [_b]
    
    for i in range(0, num_steps):
        _y_hat = predict_with_bias(x, _w, _b)
        
        _w -= lr * d_rss_w(y, _y_hat, x) / x.shape[0]
        _b -= lr * d_rss_w_0(y, _y_hat) / x.shape[0]

        _w_trail_path.append(lr * d_rss_w(y, _y_hat, x) / x.shape[0])
        _b_trail_path.append(lr * d_rss_w_0(y, _y_hat) / x.shape[0])
        
    return _w, _b, _w_trail_path, _b_trail_path
