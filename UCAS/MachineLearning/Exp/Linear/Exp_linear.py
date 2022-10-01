from base64 import b16decode
from cgi import test
from configparser import Interpolation
from operator import irshift
import os
import re
import struct
import matplotlib.pyplot as plt
from symbol import parameters
from matplotlib.font_manager import FontProperties
import numpy as np
from Exp_learn import sigmoid , softmax, dense_to_one_hot, tanh
"""单个选取训练数据处理"""

def load_mnist(path, kind='train',normal=False, onehot=False):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>2I',lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
        if onehot:
            labels = dense_to_one_hot(labels,10)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">4I",imgpath.read(16))
        if normal:
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels),784)/255
        else:
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    
    return images, labels

##reload Data
x_train, y_train = load_mnist(r"./data/MNIST/raw", kind='train', normal=True)
x_test, y_test = load_mnist(r'./data/MNIST/raw', kind='t10k',normal=True)


#初始化参数
def initialize_with_zeros(n_x,n_h,n_y,std=0.001):
    W1 = np.random.randn(n_h,n_x)*std
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*std
    b2 = np.zeros((n_y,1))
    parameters = {"W1": W1,
          "b1": b1,
          "W2": W2,
          "b2": b2}
    return parameters

#构建神经网络
def forward(X,parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    Z1 = np.dot(W1,X) + b1
    A1 = tanh(Z1)
    Z2 = np.dot(W2,A1)
    A2 = sigmoid(Z2)
    dict = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2}
    return A2, dict

#交叉熵损失函数
def loss(A2,Y,parameters):
    t=1e-6
    logprobs=np.multiply(np.log(A2+t),Y) + np.multiply(np.log(1-A2+t),(1-Y))
    loss1=np.sum(logprobs,axis=0,keepdims=True)/A2.shape[0]
    return loss1*(-1)

#反向传播函数
def backward(parameters,dict,X,Y):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = dict["A1"]
    A2 = dict["A2"]
    Z1 = dict["Z1"]
    dZ2 = A2-Y
    dW2 = np.dot(dZ2,A1.T)
    db2 = np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1,2))
    dW1 = np.dot(dZ1,X.T)
    db1 = np.sum(dZ1,axis=1,keepdims=True)
    grads = {"dW1": dW1,
         "db1": db1,
         "dW2": dW2,
         "db2": db2} 
    return grads

#梯度下降更新参数
def gradient(parameters, grads, learning_rate):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    lr = learning_rate
    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2}
    return parameters

#train model
from time import time
if __name__ == '__main__':
    train_images = x_train
    train_labels = y_train
    test_images  = x_test
    test_labels  = y_test

    count = 0
    n_x   = 28 * 28
    n_h   = 100
    n_y   = 10
    lr    = 0.01
    loss_all   = []
    train_size = 100
    n = 10000
    time_total = time()
    parameters = initialize_with_zeros(n_x, n_h, n_y)
    for i in range(n):
        time0 = time()
        img_train    = train_images[i]
        label_train1 = train_labels[i]
        label_train  = np.zeros((10,1))

        #动态修改学习率
        if i%1000 == 0:
            lr = lr * 0.99

        #转化为one-hot格式
        label_train[int(train_labels[i])] = 1
        imgvector = np.expand_dims(img_train,axis=1)
        A2, dict  = forward(imgvector, parameters)
        pre_label = np.argmax(A2)

        #统计loss
        loss1 = loss(A2 , label_train, parameters)
        grads = backward(parameters, dict, imgvector, label_train)
        parameters = gradient(parameters, grads, learning_rate = lr)
        grads["dW1"] = 0
        grads["dW2"] = 0
        grads["db1"] = 0
        grads["db2"] = 0

        if i%200==0:
            print("迭代：{} 次的损失值:{:.6f}, 耗时: {} s".format(i,loss1[0][0],(time()-time0)*1000))
            loss_all.append(loss1[0][0])
            time0 = 0
        
        #训练模型
    for i in range(n):
        img_test = test_images[i]
        vector_image = np.expand_dims(img_test, axis=1)
        label_trainx = test_labels[i]
        aa2, xxx = forward(vector_image,parameters)
        predict_value = np.argmax(aa2)
        if predict_value == int(label_trainx):
            count += 1
            
    print("准确率：{} , 共耗时：{} s".format(count / n, (time()-time_total)))
    plt.figure()
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.plot(range(0,10000,200),loss_all)
    plt.xlabel("迭代次数")
    plt.ylabel("损失率")
    plt.show()


        

