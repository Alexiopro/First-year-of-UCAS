import imp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt

#载入图像数据
from keras.utils.image_utils import load_img, save_img, img_to_array
base_image_path = os.path.join("./data/style_transfer/content/content1.jpg")
style_reference_image_path = os.path.join("./data/style_transfer/style/autumn_girl.jpg")
width, height = load_img(base_image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height )

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.applications import vgg19
tf.compat.v1.disable_eager_execution()

#预处理函数
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

#处理数据转图像
def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

#定义loss函数
#Gram Matrix 计算
def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

#风格损失
def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    G = gram_matrix(combination)
    channels = 3
    size = img_ncols * img_nrows
    return K.sum(K.square( S - G)) / (4.0 * (channels**2) * (size**2))

#内容损失
def content_loss(base, combination):
    return K.sum(K.square( combination - base))

#Variable loss
def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

#读取并预处理原始数据
base_image = K.variable(preprocess_image(base_image_path))
style_reference_image = K.variable(preprocess_image(style_reference_image_path))
# 使用placeholder定义生成图像
if K.image_data_format() == 'channels_first':
    combination_image = K.placeholder((1, 3, img_nrows, img_ncols))
else:
    combination_image = K.placeholder((1, img_nrows, img_ncols, 3))
# 把内容图像C,风格图像C以及生成图像G连接，组成VGG19模型的输入矩阵
input_tensor = K.concatenate(
    [base_image,
    style_reference_image,
    combination_image], axis=0)

#载入VGG19模型
model = vgg19.VGG19(input_tensor=input_tensor, 
        weights='imagenet', include_top=False)

#提取各个神经层
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

#总体损失函数是三个损失部分的加权和
total_variation_weight = 1.0
style_weight = 1.0
content_weight = 0.025

loss = K.variable(0.0)

#首先计算内容损失
layer_features = outputs_dict['block5_conv2']
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss = loss + content_weight * content_loss(base_image_features, combination_features)
#定义在那些卷积层上提取风格特征，分别计算风格损失
feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss = loss + (style_weight / len(feature_layers)) * sl

#添加整体波动损失以保证生成图像的平滑性
loss = loss + total_variation_weight * total_variation_loss(combination_image)

#计算损失函数对于生成图像的梯度
grads = K.gradients(loss, combination_image)[0]

# Optimizer Create
outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_image], outputs)
# loss and grads function create
def eval_loss_grads(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

# loss and grads calculation
class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grad_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value
    
    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()
epochs = 500
result_prefix = './data/style_transfer/combinations/result_'

#开始风格迁移
import time
from scipy.optimize import fmin_l_bfgs_b
x = preprocess_image(base_image_path)
start_time = time.time()
for i in range(epochs):
    print('Epochs %d' % (i+1))
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                    fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    img = deprocess_image(x.copy())
    fname = result_prefix + '_at_iteration_%d.png' % (i+1)
    save_img(fname, img)
    end_time = time.time()
    print('Imgae saved as', fname)
print('Iteration %d completed in %ds' % (i+1, end_time - start_time))

from PIL import Image

plt.figure() 
for i in range(0,10):
    plt.subplot(5,2,i+1)
    plt.title("ok"+str(i+1))
    img=Image.open('./data/style_transfer/combinations/result__at_iteration_'+str(i+1)+'.png')
    plt.imshow(img)
#plt.show()

