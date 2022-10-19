import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# 设置随机数种子(自动炼丹)
import torch
#num = np.random.choice(100)
#print(num)
torch.manual_seed(11)


S1 = [[ 1.58, 2.32, -5.80], [ 0.67, 1.58, -4.78], [ 1.04, 1.01, -3.63],
[-1.49, 2.18, -3.39], [-0.41, 1.21, -4.73], [ 1.39, 3.16,  2.87],
[ 1.20, 1.40, -1.89], [-0.92, 1.44, -3.22], [ 0.45, 1.33, -4.38],
[-0.76, 0.84, -1.96]]

S2 = [[ 0.21, 0.03, -2.21], [ 0.37, 0.28, -1.8], [ 0.18, 1.22, 0.16], 
[-0.24, 0.93, -1.01], [-1.18, 0.39, -0.39], [0.74, 0.96, -1.16],
[-0.38, 1.94, -0.48], [0.02, 0.72, -0.17], [ 0.44, 1.31, -0.14],
[ 0.46, 1.49, 0.68]]

S3 = [[-1.54, 1.17, 0.64], [5.41, 3.45, -1.33], [ 1.55, 0.99, 2.69], 
[1.86, 3.19, 1.51], [1.68, 1.79, -0.87], [3.51, -0.22, -1.39],
[1.40, -0.44, -0.92], [0.44, 0.83, 1.97], [ 0.25, 0.68, -0.99],
[ 0.66, -0.45, 0.08]]

def gen_train_data(data_input):
    train_data = []
    train_label = []
    for idx, i in enumerate(data_input):
        for j in i:
            data = np.array(j)
            train_data.append(data)
            # trans to one-hot
            label = np.zeros_like(data)
            label[idx] = 1
            train_label.append(label)
    return train_data, train_label

data, label = gen_train_data([S1, S2, S3])
train_data  = tf.convert_to_tensor(data)
train_label = tf.convert_to_tensor(label)

###define train_data and train_label with first five elements in data and label
#train_data  = tf.convert_to_tensor(np.vstack((data[:5],data[10:15],data[20:25])))
#train_label = tf.convert_to_tensor(np.vstack((label[:5],label[10:15],label[20:25])))
##define test_data and test_label
#test_data   = tf.convert_to_tensor(np.vstack((data[5:10],data[15:20],data[25:])))
#test_label  = tf.convert_to_tensor(np.vstack((label[5:10],label[15:20],label[25:])))

loss , accuracy = [],[]
#define optimizers parameters matrix
hd    = [3, 6, 9]          # number of Hidden node
lr    = [0.1, 0.2, 0.3]     # learning_rate
epoch = [100, 200, 300]     # recycle num
batch_size = 3  
verbose = 0

#define optimizers term
sgd1 = SGD(learning_rate=lr[0], momentum=0.99)
sgd2 = SGD(learning_rate=lr[1], momentum=0.99)
sgd3 = SGD(learning_rate=lr[2], momentum=0.99)
#build network layers Q1
#Hidden node = 3
model11 = Sequential()
model11.add(Dense(hd[0], activation='tanh', input_dim=3))
model11.add(Dense(3,activation='sigmoid'))
model11.compile(optimizer= sgd1, loss='mean_squared_error', metrics=['accuracy'])

#Hidden node = 6
model12 = Sequential()
model12.add(Dense(hd[1], activation='tanh', input_dim=3))
model12.add(Dense(3,activation='sigmoid'))
model12.compile(optimizer= sgd1, loss='mean_squared_error', metrics=['accuracy'])

#Hidden node = 9
model13 = Sequential()
model13.add(Dense(hd[2], activation='tanh', input_dim=3))
model13.add(Dense(3,activation='sigmoid'))
model13.compile(optimizer= sgd1, loss='mean_squared_error', metrics=['accuracy'])

#build network layers Q2
#Learning rate = 0.1
model21 = Sequential()
model21.add(Dense(hd[1], activation='tanh', input_dim=3))
model21.add(Dense(3,activation='sigmoid'))
model21.compile(optimizer= sgd1, loss='mean_squared_error', metrics=['accuracy'])

#Learning rate = 0.2
model22 = Sequential()
model22.add(Dense(hd[1], activation='tanh', input_dim=3))
model22.add(Dense(3,activation='sigmoid'))
model22.compile(optimizer= sgd2, loss='mean_squared_error', metrics=['accuracy'])

#Learning rate = 0.3
model23 = Sequential()
model23.add(Dense(hd[1], activation='tanh', input_dim=3))
model23.add(Dense(3,activation='sigmoid'))
model23.compile(optimizer= sgd3, loss='mean_squared_error', metrics=['accuracy'])

#build network layers Q2
#Learning rate = 0.1
model21 = Sequential()
model21.add(Dense(hd[1], activation='tanh', input_dim=3))
model21.add(Dense(3,activation='sigmoid'))
model21.compile(optimizer= sgd1, loss='mean_squared_error', metrics=['accuracy'])

#Learning rate = 0.2
model22 = Sequential()
model22.add(Dense(hd[1], activation='tanh', input_dim=3))
model22.add(Dense(3,activation='sigmoid'))
model22.compile(optimizer= sgd2, loss='mean_squared_error', metrics=['accuracy'])

#Learning rate = 0.3
model23 = Sequential()
model23.add(Dense(hd[1], activation='tanh', input_dim=3))
model23.add(Dense(3,activation='sigmoid'))
model23.compile(optimizer= sgd3, loss='mean_squared_error', metrics=['accuracy'])

#train the models with batch_size=3 \\ Q1
hist11 = model11.fit(train_data, train_label,epochs=epoch[0],
                 verbose=verbose, batch_size = batch_size  )

hist12 = model12.fit(train_data, train_label,epochs=epoch[0],
                 verbose=verbose, batch_size = batch_size  )

hist13 = model13.fit(train_data, train_label,epochs=epoch[0],
                 verbose=verbose, batch_size = batch_size  )

#train the models with batch_size=3 \\ Q2
hist21 = model21.fit(train_data, train_label,epochs=epoch[0],
                 verbose=verbose, batch_size = batch_size  )

hist22 = model22.fit(train_data, train_label,epochs=epoch[0],
                 verbose=verbose, batch_size = batch_size  )

hist23 = model23.fit(train_data, train_label,epochs=epoch[0],
                 verbose=verbose, batch_size = batch_size  )

#train the models with batch_size=3 \\ Q3
hist3 = model12.fit(train_data, train_label,epochs=epoch[0],
                 verbose=verbose, batch_size = 1  )

#save the loss and accuracy
acc11  = hist11.history['accuracy']
loss11 = hist11.history['loss']
acc12  = hist12.history['accuracy']
loss12 = hist12.history['loss']
acc13  = hist13.history['accuracy']
loss13 = hist13.history['loss']
acc21  = hist21.history['accuracy']
loss21 = hist21.history['loss']
acc22  = hist22.history['accuracy']
loss22 = hist22.history['loss']
acc23  = hist23.history['accuracy']
loss23 = hist23.history['loss']
loss3 = hist12.history['loss']
# draw model
import matplotlib.pyplot as plt

epochs = range(1, len(acc12) + 1)
plt.plot(epochs, loss21,  'r',  label='$\eta$=0.1')
plt.plot(epochs, loss22,  'g',  label='$\eta$=0.2')
plt.plot(epochs, loss23,  'b',  label='$\eta$=0.3')
plt.title('Loss in different epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss ')
plt.legend()
plt.show()
