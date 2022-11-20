import numpy as np
import matplotlib.pyplot as plt

def make_seed(SEED=2):
    np.random.seed(SEED)

## calculate the distance between sample points and the center of sample
## return the center mark corresponding to the shortest distance
def distance(samples, center):
    l = np.power(samples - center, 2).sum(axis=1)
    return np.argmin(l)

## visualization module
def clusters_show(clusters,centers, step):
    color = ["red", "blue", "green","yellow"]
    marker = ["*", "^", "D","X"]
    label  = ["Category I", "Category II", "Category III", "Category IV"]
    plt.figure(figsize=(8, 8))
    plt.title("step: {}".format(step))
    plt.xlabel("Density", loc="center")
    plt.ylabel("Sugar Content", loc="center")

    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1], c=color[i],marker=marker[i],s=80)
        plt.legend(label,loc='upper left')
    for j in range(len(centers)):
        plt.scatter(centers[j][0],centers[j][1],marker="+", c="black")
    #plt.show()

## computing cluster center computing module
def compute_center(samples,center,n):
    step = 0
    while True:
        sign = 0
        clusters = [[] for i in range(n)]
        for sample in samples:
            label = distance(sample, center)
            clusters[label].append(sample)
        print(clusters)
        #visualization
        clusters_show(clusters,center, step)
        for labels, values in enumerate(clusters):
            next_center = np.array(values).mean(axis=0)

            if (center[labels] != next_center).all() :
                center[labels] = next_center
            else:
                sign += 1   
        step += 1

        print("step:{}".format(step), "\n", "centers:{}".format(center))
        ## Judge whether to follow the new cluster center value.
        ## If the cluster center does not change in two adjacent iterations, the iteration is terminated.
        if sign > 3:
            break

    return center, step

##Obtain data belonging to their respective categories according to the final cluster center
def split_data(samples, center):
    n = len(samples)
    clusters = [[] for i in range(n)]
    for sample in samples:
        label = distance(sample, center)
        clusters[label].append(sample)

    return clusters

if __name__ == '__main__':

    make_seed()
    k = 4 ##Cluster number
    samples=np.array([
    [0.697,0.460],[0.774,0.376],[0.634,0.264],[0.608,0.318],[0.556,0.215],
    [0.403,0.237],[0.481,0.149],[0.437,0.211],[0.666,0.091],[0.243,0.267],
    [0.245,0.057],[0.343,0.099],[0.639,0.161],[0.657,0.198],[0.360,0.370],
    [0.593,0.042],[0.719,0.103],[0.359,0.188],[0.339,0.241],[0.282,0.257],
    [0.748,0.232],[0.714,0.346],[0.483,0.312],[0.478,0.437],[0.525,0.369],
    [0.751,0.489],[0.532,0.472],[0.473,0.376],[0.725,0.445],[0.446,0.459]])
    center = samples[np.random.choice(len(samples), k, replace=False)]
    print("初始聚类中心为:",center,sep="\n")
    centers, step = compute_center(samples,center,k)
    clusters = split_data(samples,centers)
    print("分类完成!共迭代%d次"%(step-1))
    print("共分为%d类,最终分类的结果为:"%(k))

    for i in range(k):
        label = []
        labels = []
        SsE = np.power(clusters[i] - centers[i], 2).sum()
        print("第 %d 类最终聚类中心为:"%(i+1),centers[i],"误差平方和为:",SsE,sep="\n")
        for j in range(len(clusters[i])):
            label.append(np.argwhere(list(clusters[i][j] == samples)))
        for k in range(len(label)):
            labels.append(label[k][0][0]+1)
        print("各类样本在原数组的编号分别是:",labels)