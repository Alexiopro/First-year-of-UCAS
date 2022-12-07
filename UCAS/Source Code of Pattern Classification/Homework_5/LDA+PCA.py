import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

def make_seed(SEED=5):
    np.random.seed(SEED)

#Data loader
from scipy.io import loadmat
class loadData(object):
    def __init__(self, orl_file='./data/ORLData_25', vehicle_file='./data/vehicle'):
        # 读取数据集
        self.orl_data = loadmat(orl_file)
        self.vehicle_data = loadmat(vehicle_file)
 
    def get_orl_data(self):
        orl_data = self.orl_data['ORLData'].T   # 400*645大小，每一行是一个样本，每一列是一维特征
        all_data = orl_data[:, 0:-1]  # 提取特征
        all_data = np.asarray(all_data).astype(float)
        all_label = orl_data[:, -1]  # 提取标签
        all_label = np.asarray(all_label).astype(float)
 
        return all_data, all_label
 
    def get_vehicle_data(self):
        UCI_entropy_data = self.vehicle_data['UCI_entropy_data']
        data = UCI_entropy_data[0, 0]['train_data']
        data = data.T   # 转置一下 变成846*19
        all_data = data[:, 0:-1]
        all_label = data[:, -1]
        all_data = np.asarray(all_data).astype(float)
        all_label = np.asarray(all_label).astype(float)

        return all_data, all_label

#K-NN classification
from collections import Counter
import math
class KNNClassifier:
    def __init__(self, k):
        assert k >= 1, "k must be valid"  # 抛出异常
        self.k = k
        self.X_train = None  # 训练数据集在类中，用户不能随意操作，故设置为私有
        self.y_train = None
        self.result = None
 
    def fit(self, X_train, y_train):
        """根据训练数据集X_train和y_train训练kNN分类器"""
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must equal to the size of y_train"
        assert self.k <= X_train.shape[0], "the size of X_train must be at least k."
        self.X_train = X_train
        self.y_train = y_train
        return self  # 模仿sklearn，调用fit函数会返回自身
 
    def predict(self, X_test):
        """给定待预测数据集X_predict, 返回结果向量"""
        assert self.X_train is not None and self.y_train is not None, "must fit before predict!"
        assert X_test.shape[1] == self.X_train.shape[1], "the feature number of X_predict must be equal to X_train"
        # 预测X_predict矩阵每一行所属的类别
        pred_result = [self._predict(x) for x in X_test]
        self.result = np.array(pred_result)
        return self.result  # 返回的结果也遵循sklearn
 
    def _predict(self, x):
        """给定单个待预测的数据x,返回x_predict的预测结果值"""
        # 先判断x是合法的
        assert x.shape[0] == self.X_train.shape[1], "the feature number of x must be equal to X_train"
        # 计算新来的数据与整个训练数据的距离
        distances = [math.sqrt(np.sum((x_train - x) ** 2)) for x_train in self.X_train]
        nearest = np.argsort(distances)  # 对距离排序并返回对应的索引
        topK_y = [self.y_train[i] for i in nearest[:self.k]]  # 返回最近的k个距离对应的分类
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]
 
    # 计算准确率
    def get_accuracy(self, y_test):
        assert self.result is not None, "must predict before calculate accuracy!"
        assert y_test.shape[0] == self.result.shape[0], "the label number of test data must be equal to train data!"
        correct = 0
        for i in range(len(self.result)):
            if y_test[i] == self.result[i]:
                correct += 1
        return (correct / float(len(self.result))) * 100.0

#PCA methods
class PCA(object):
    """
    components: Proportion of principal components
    """
    def __init__(self, x, components):
        self.x = x
        self.components = components
        self.m, self.n = np.shape(x)  # 记下降维矩阵的样本数m和特征维度n
        self.pcaData = None
 
    def get_average(self):
        average = np.mean(self.x, axis=0)  # 对数组第0个维度求均值，就是求每列的均值 得到每个特征的平均 1*n
        # average = np.tile(average, (self.m, 1))  # 得到均值矩阵 m*n
        return average
 
    def zero_mean(self, ave):
        zero_mean = self.x - ave
        return zero_mean
 
    def start_pca(self):
        ave = self.get_average()
        zero_m = self.zero_mean(ave)
        covX = np.cov(zero_m, rowvar=False)  
        eigVals, eigVects = np.linalg.eig(covX)  # 求解协方差矩阵的特征值和特征向量
        index = np.argsort(-eigVals)  # 按照特征值进行从大到小排序
        eigVals_ = eigVals[index]  # 得到重新排序的特征值
        eigVals_ = np.asarray((eigVals_).real).astype(float)
        featSum = np.sum(eigVals_)
        featSum_ = 0.0
        k = 0
 
        if 0 < self.components < 1:  # 如果传进来的参数是特征占比（默认0.9）,计算所需要保留的维度,直到保留特征占比到设定值为止
            for i in range(self.n):
                featSum_ = featSum_ + eigVals_[i]
                k = k + 1
                if featSum_ / featSum > self.components:
                    break
            #print('保留的特征占比为', self.components)
            #print('保留了前%d个维度' % k)
        if self.components >= 1:
            assert self.components % 1 == 0, "if proportion≥1,it equals to your target dimension! It must be integer!"
            featSum_ = 0.0
            k = self.components
            for i in range(k):
                featSum_ = featSum_ + eigVals_[i]
            proportion = featSum_ / featSum * 100
            #print('保留的特征占比为', proportion)
            #print('保留了前%d个维度' % k)
 
        selectVec = np.mat(eigVects[:, index[:k]])  # 选出前k个特征向量，保存为矩阵
 
        final_data = zero_m * selectVec  # 低维特征空间的数据
        #reconData = (final_data * selectVec.T) + ave  # 重构数据
        finalData_ = np.asarray((final_data).real).astype(float)
        #reconMat_ = np.asarray((reconData).real).astype(float)
        self.pcaData = finalData_
        return finalData_,proportion  # 返回降维以后的数据、重构以后的数据

#LDA methods
class LDA(object):
    """
    n_dim: Maintain the dimension of the feature
    """
    def __init__(self, data, label, n_dim):
        self.X = data
        self.y = label
        self.n_dim = n_dim
        self.clusters = np.unique(label)
        self.lda_data = None
        assert n_dim < len(self.clusters), "your target dimension is too big!"
        assert len(self.X) == len(self.y), "the length of label must be equal to your data"
 
    def zero_mean(self, x):
        average = np.mean(x, axis=0)  # 对数组第0个维度求均值，就是求每列的均值 得到每个特征的平均 1*n
        zero_mean = x - average
        return zero_mean
 
    def get_Sw(self):  # 求类内散度矩阵
        Sw = np.zeros((self.X.shape[1], self.X.shape[1]))  # 初始化散度矩阵
        for i in self.clusters:  # 对每个类别分别求类内散度后相加
            data_i = self.X[self.y == i]
            Swi = np.mat(self.zero_mean(data_i)).T * np.mat(self.zero_mean(data_i))
            Sw += Swi
        return Sw
 
    def get_Sb(self):  # 求类间散度矩阵，即全局散度-类内散度
        temp = np.mat(self.zero_mean(self.X))
        St = temp.T * temp
        Sb = St - self.get_Sw()
        return Sb
 
    def start_lda(self):
        Sb = self.get_Sb()
        Sw = self.get_Sw()
        S = np.linalg.inv(Sw) * Sb  # 计算矩阵Sw^-1*Sb
        eigVals, eigVects = np.linalg.eig(S)  # 求S的特征值，特征向量
        index = np.argsort(-eigVals)  # 按照特征值进行从大到小排序

        # 计算保留的特征占比
        eigVals_ = eigVals[index]  # 得到重新排序后的特征值
        eigVals_ = np.asarray(abs(eigVals_)).astype(float)
        featSum = np.sum(eigVals_)

        featSum_ = 0.0
        for i in range(self.n_dim):
            featSum_ = featSum_ + eigVals_[i]
        proportion = featSum_ / featSum * 100
        #print('the proportion of remained feature is:', proportion)

        w = np.mat(eigVects[:, index[:self.n_dim]])  # 选出前n_dim个特征向量，保存为矩阵
        data_ndim = np.asarray((self.X * w).real).astype(float)

        self.lda_data = data_ndim
        return data_ndim

if __name__ == '__main__':
    make_seed()
    ld = loadData()
    origin_orl_data, orl_label = ld.get_orl_data()
    origin_vehicle_data, vehicle_label = ld.get_vehicle_data()

    m = 50
    Lda_orl_acc = []
    Pca_orl_acc = []
    Lda_vehicle_acc = []
    Pca_vehicle_acc = []

    for i in range(m):
        if i == 0:
            classify_orl_data = True
            classify_vehicle_data = True
            continue

        if i == len(np.unique(orl_label)):
            classify_orl_data = False
            continue

        if classify_orl_data :
            lda_ = LDA(origin_orl_data, orl_label, 40-i)
            orl_data = lda_.start_lda()
            train_data, val_data, train_label, val_label = train_test_split(
                orl_data, orl_label, test_size=0.2,random_state=1, stratify=orl_label)
            
            knn_cf = KNNClassifier(1)
            knn_cf.fit(train_data, train_label)
            knn_cf.predict(val_data)
            acc = knn_cf.get_accuracy(val_label)
            Lda_orl_acc.append([i, acc])
        
        if i == len(np.unique(vehicle_label)):
            classify_vehicle_data = False
            continue

        if classify_vehicle_data:
            lda_ = LDA(origin_vehicle_data, vehicle_label, i)
            vehicle_data = lda_.start_lda()
            train_data, val_data, train_label, val_label = train_test_split(
            vehicle_data, vehicle_label, test_size=0.2,random_state=1, stratify=vehicle_label)
            knn_cf = KNNClassifier(1)
            knn_cf.fit(train_data, train_label)
            knn_cf.predict(val_data)
            acc = knn_cf.get_accuracy(val_label)
            Lda_vehicle_acc.append([i, acc])

    for j in range(m):
        if j == 0:
            classify_orl_data = True
            classify_vehicle_data = True
            continue
 
        if j == len(np.unique(orl_label)):
            classify_orl_data = False
            continue
 
        if classify_orl_data:
            pca_ = PCA(origin_orl_data,j)
            orl_data,pac_orl_proportion = pca_.start_pca()
            train_data, val_data, train_label, val_label = train_test_split(
                orl_data, orl_label, test_size=0.2, random_state=1, stratify=orl_label)
            knn_cf = KNNClassifier(1)   # 最近邻分类
            knn_cf.fit(train_data, train_label)
            knn_cf.predict(val_data)
            acc = knn_cf.get_accuracy(val_label)
            Pca_orl_acc.append([j, acc,pac_orl_proportion])

        if j == 18:
            classify_vehicle_data = False
            continue

        if classify_vehicle_data:
            pca_ = PCA(origin_vehicle_data,j)
            vehicle_data,pac_vehicle_proportion = pca_.start_pca()
            train_data, val_data, train_label, val_label = train_test_split(
            vehicle_data, vehicle_label, test_size=0.2,random_state=1, stratify=vehicle_label)
            knn_cf = KNNClassifier(1)
            knn_cf.fit(train_data, train_label)
            knn_cf.predict(val_data)
            acc = knn_cf.get_accuracy(val_label)
            Pca_vehicle_acc.append([j, acc,pac_vehicle_proportion])

    Lda_orl_acc = np.asarray(Lda_orl_acc)
    Lda_vehicle_acc = np.asarray(Lda_vehicle_acc)
    Pca_orl_acc = np.asarray(Pca_orl_acc)
    Pca_vehicle_acc = np.asarray(Pca_vehicle_acc)

    plt.figure(figsize=(8, 4))
    plt.rcParams['font.sans-serif'] = ['SimHei']
 
    plt.subplot(121)
    plt.title("orl_LDA")
    plt.plot(Lda_orl_acc[:, 0], Lda_orl_acc[:, 1], label="正确率")
    plt.xlabel('保留特征维度')
    plt.legend()
    plt.grid()
    
    plt.subplot(122)
    plt.title("vehicle_LDA")
    plt.plot(Lda_vehicle_acc[:, 0], Lda_vehicle_acc[:, 1], label="正确率")
    plt.xlabel('保留特征维度')
    plt.legend()
    plt.grid()

    plt.figure(figsize=(8, 4))

    plt.subplot(121)
    plt.title("orl_PCA")
    plt.plot(Pca_orl_acc[:, 0], Pca_orl_acc[:, 1], label="正确率")
    plt.plot(Pca_orl_acc[:, 0], Pca_orl_acc[:, 2], label="特征占比")
    plt.xlabel('保留特征维度')
    plt.legend()
    plt.grid()
    
    plt.subplot(122)
    plt.title("vehicle_PCA")
    plt.plot(Pca_vehicle_acc[:, 0], Pca_vehicle_acc[:, 1], label="正确率")
    plt.plot(Pca_vehicle_acc[:, 0], Pca_vehicle_acc[:, 2], label="特征占比")
    plt.xlabel('保留特征维度')
    plt.legend()
    plt.grid()
 
    plt.show()


            