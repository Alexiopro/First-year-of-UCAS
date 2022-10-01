import matplotlib.pyplot as plt
import numpy as np
from pip import main
 
x = np.loadtxt('data.txt')  ##data数据已经规范化增广处理
b = np.ones(40)
y = np.insert(x, 0, b, 1)  # 增广
# 创立 BatchPerception 类
class BatchPerception():
    def __init__(self, w1, w2, y):
        self.w1 = w1
        self.w2 = w2
        self.a = np.zeros(3)
        self.epochs = 0
        self.lr = 1
        self.y = y
 
    def preprocess(self):
        y_temp = self.y.copy()
        y_w1 = y_temp[(self.w1 - 1) * 10:self.w1 * 10, 0:3]
        y_w2 = -1 * y_temp[(self.w2 - 1) * 10:self.w2 * 10, 0:3] # 规范化
        y_w = np.concatenate((y_w1, y_w2), axis=0)
        return y_w
 
    def train(self):
        y_w = self.preprocess()
        for j in range(1000):
            Y = []    # 损失函数
 
            for i in range(20):
                if np.inner(self.a, y_w[i]) <= 0:
                    Y.append(y_w[i])
            if len(Y) == 0:
                print("Total iterations: %d" %self.epochs)
                print("w{} and w{}'s Weight vectors is: {}".format(self.w1, self.w2, self.a))
                break
            Y_sum = np.sum(Y, axis=0)
            self.a = self.a + self.lr * Y_sum   # SGD optimizer
            self.epochs += 1
 
 # 可视化模组
    def visualization(self):
        y_temp = self.y.copy()
        y_w1 = y_temp[(self.w1 - 1) * 10:self.w1 * 10, 0:3]
        y_w2 = y_temp[(self.w2 - 1) * 10:self.w2 * 10, 0:3]
        y = np.concatenate((y_w1, y_w2), axis=0)
        ax = plt.gca()
        plt.scatter(y[0:10, 1], y[0:10, 2], s=16, label='sample of w' + str(self.w1))
        plt.scatter(y[10:20, 1], y[10:20, 2], s=16, label='sample of w' + str(self.w2))
        x_min = np.min(y[:, 1])
        x_max = np.max(y[:, 1])
        x_plot = np.arange(x_min, x_max, (x_max - x_min) / 100)
        y_plot = -(self.a[1] * x_plot + self.a[0]) / self.a[2]
        plt.plot(x_plot, y_plot, linewidth=2, label='boundary', color='red')
        plt.legend()
        plt.title('classification result by BatchPerception between $\omega$' + str(self.w1) + ' and $\omega$' + str(self.w2))
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        plt.savefig('BP'+str(self.w1)+str(self.w2)+'.png')
        plt.show()
 
 #创立 MSE 类
class MSE_Expand():
    def __init__(self, y):
        self.y = y
        self.a = np.zeros([4, 3]) #4*3 * 3*32
        self.label_train = np.zeros([4, 32])
        self.label_test = np.zeros([4, 8])
 
    def preprocess(self):
        y_temp = self.y.copy()

        # Train[0:8] and Test[8:10] gather
        y_train = np.concatenate((y_temp[0:8, 0:3], y_temp[10:18, 0:3], y_temp[20:28, 0:3], y_temp[30:38, 0:3]), axis=0).T  # 3*32
        y_test = np.concatenate((y_temp[8:10, 0:3], y_temp[18:20, 0:3], y_temp[28:30, 0:3], y_temp[38:40, 0:3]), axis=0).T
        for i in range(len(self.label_train)):
            for j in range(len(self.label_train[0])):
                self.label_train[i, j] = int(int(j/8)==i)
        for i in range(len(self.label_test)):
            for j in range(len(self.label_test[0])):
                self.label_test[i, j] = int(int(j/2)==i)
        return y_train, y_test

    def train(self):
        y_train, y_test = self.preprocess()
 
        a_temp = np.matmul(np.linalg.pinv(y_train.T), self.label_train.T)
        self.a = a_temp.T
        self.test(y_test)
 
 #测试函数
    def test(self, y_test):
        t = np.arange(1,5)
        test = np.matmul(t, self.label_test)
        result = np.argmax(np.matmul(self.a, y_test), axis=0)+np.ones_like(np.argmax(np.matmul(self.a, y_test), axis=0))
        correct = sum(test == result) / len(self.label_test[0])
        print("Total correct of MSE_expand methods is: %d" %correct)
 
#主程序
if __name__ == "__main__":
    BP_12 = BatchPerception(1, 2, y)
    BP_34 = BatchPerception(3, 4, y)
    BP_12.train()
    BP_34.train()
    BP_12.visualization()
    BP_34.visualization()
    ME = MSE_Expand(y)
    ME.train()

    