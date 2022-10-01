from pickletools import optimize
from tensorflow import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

x_train = np.random.random((1000, 784))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((200, 784))
y_test = np.random.randint(2, size=(200, 1))

#构建网络
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=784))
model.add(Dense(1, activation='sigmoid'))

#定义优化器、损失函数
model.compile(optimizer= 'Adam', loss='binary_crossentropy', metrics=['accuracy'])

#训练数据，verbose：训练时显示实时信息，0表示不显示数据，1表示显示进度条，2表示用只显示一个数据。
model.fit(x_train, y_train, epochs=10, verbose=0, batch_size=32 )

#测试模型准确性
score = model.evaluate(x_test, y_test, batch_size=32)
print("模型准确率是:{:.2f} %".format(score[1]*100))

json_string = model.to_json()  

#从保存的json中加载模型  
from keras.models import model_from_json, load_model
model_re = model_from_json(json_string)

#保存模型
model.save('MachineLearning\keras_model\my_mode101.h5')
#加载模型
re_model = load_model('MachineLearning\keras_model\my_mode101.h5')