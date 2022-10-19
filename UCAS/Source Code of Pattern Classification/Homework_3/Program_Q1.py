import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import torch
torch.manual_seed(3047)
#reload the database AND label them

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
#define optimizers parameters
hd    = 3       # number of Hidden layers
lr    = 0.1    # learning_rate
epoch = 80     # recycle num
batch_size = 3  
#define optimizers term
sgd = SGD(learning_rate=lr, momentum=0.99)

#build network layers
model = Sequential()
model.add(Dense(hd, activation='tanh', input_dim=3))
model.add(Dense(3,activation='sigmoid'))
model.compile(optimizer= sgd, loss='mean_squared_error', metrics=['accuracy'])

#train the models with batch_size=3 
import keras
hist1 = model.fit(train_data, train_label,epochs=epoch,
                verbose=2, batch_size = batch_size  )

#resetting the parameters
for ix, layer in enumerate(model.layers):
    if hasattr(model.layers[ix], 'kernel_initializer') and \
            hasattr(model.layers[ix], 'bias_initializer'):
        weight_initializer = model.layers[ix].kernel_initializer
        bias_initializer = model.layers[ix].bias_initializer

        old_weights, old_biases = model.layers[ix].get_weights()

        model.layers[ix].set_weights([
            weight_initializer(shape=old_weights.shape),
            bias_initializer(shape=len(old_biases))])

#train the models with none_batch
hist2 = model.fit(train_data, train_label,
                epochs=epoch, verbose=2, batch_size = 1)

# draw model
import matplotlib.pyplot as plt

acc1  = hist1.history['accuracy']
loss1 = hist1.history['loss']

acc2  = hist2.history['accuracy']
loss2 = hist2.history['loss']

epochs = range(1, len(acc1) + 1)
plt.plot(epochs, acc2,  'r',  label='acc')
plt.plot(epochs, acc1,  'b',  label='acc: batch_size = 3')
plt.plot(epochs, loss2, 'g',  label='loss')
plt.plot(epochs, loss1, 'y',  label='loss: batch_size = 3')
plt.title('Training')
plt.xlabel('Epochs')
plt.ylabel('Accuracy and Loss')
plt.legend()
plt.show()
