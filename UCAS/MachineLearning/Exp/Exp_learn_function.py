from cmath import log
from re import X
import numpy as np
import torch
""" Basic Machine Learning Equation includes sigmoid ,relu and tanh
LeaRn
=====
"""
def sigmoid(n):
    """Returns the activation function about(1+e^(-x)^-1)"""
    return 1/(1+np.exp(-n))

def relu(X):
    """Returns the activation function about bigger one from x and 0"""
    return np.maximum(0,X)

def tanh(x):
    """Returns the activation function about tanh(x)"""
    exp2x = np.exp(-2*x)
    return (1 - exp2x)/(1 + exp2x)

def softmax(x):
    """Returns the probability of array X"""
    max_x = np.max(x)
    x = x - max_x
    sum_expX = np.sum(np.exp(x))
    return np.exp(x)/sum_expX

def mse_loss(y,t):
    """Returns the Standard Deviation of array y and t(true ans)"""
    return 0.5 * np.sum((y-t)**2)

def ce_loss(y,t):
    """Returns the Cross Entropy Deviation of array y and t(true ans)"""
    d = 1e-10
    return -np.sum(t*np.log(y+d))

def dense_to_one_hot(labels_dense, num_classes):
  """将类标签从标量转换为一个one-hot向量"""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot
