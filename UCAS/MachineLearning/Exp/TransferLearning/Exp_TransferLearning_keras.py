from email.mime import application
import os

from tensorflow.python import test
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#导入VGG模型
from keras.applications.vgg16 import VGG16

img_width , img_height = 150, 150
model = VGG16(weights='imagenet', include_top=False, input_shape= (img_width, img_height, 3))
#冻结前10层
for layer in model.layers[:11]:
    layer.trainable = False

#添加输出（分类）层
from keras.models import Sequential, Model
from keras import layers

x = model.output
x = layers.Flatten()(x)
x = layers.Dense(4096, activation='relu')(x)
x = layers.Dense(1024, activation='relu')(x)
predictions = layers.Dense(1, activation='sigmoid')(x)
model_final = Model(inputs = model.input, outputs = predictions)
model_final.summary()

#编译优化与损失函数
from keras import optimizers
model_final.compile(loss = 'binary_crossentropy',
            optimizer = optimizers.SGD(learning_rate = 0.0001, momentum = 0.99),
            metrics= ['accuracy'])

# 初始化训练集和测试集，并使用图像增强方法
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)
test_datagen = ImageDataGenerator(rescale = 1./255)  

#导入数据
train_data_dir = './data/cats_and_dogs_small/train'
validation_data_dir = './data/cats_and_dogs_small/validation'
test_data_dir = './data/cats_and_dogs_small/test'
batch_size = 20

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size, 
    class_mode = "binary")
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_height, img_width),
    class_mode = "binary")  

#train
fit_history = model_final.fit(
    train_generator,
    steps_per_epoch = 100,
    epochs= 30,
    validation_data= validation_generator,
    validation_steps= 32
)

import matplotlib.pyplot as plt
plt.figure(1, figsize = (15,8)) 

plt.subplot(221)  
plt.plot(fit_history.history['accuracy'])  
plt.plot(fit_history.history['val_accuracy'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 

plt.subplot(222)  
plt.plot(fit_history.history['loss'])  
plt.plot(fit_history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 

plt.show()