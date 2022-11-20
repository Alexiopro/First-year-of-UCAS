import numpy as np
import matplotlib.pyplot as plt

global MAX 
MAX = 10000.0
global Epsilon
Epsilon = 1e-7

def make_seed(SEED=10):
    np.random.seed(SEED)

## calculate the distance between sample points and the center of sample
## return the center mark corresponding to the shortest distance
def distance(samples, center):
    l = np.power(samples - center, 2).sum()
    return l

def Distance(M):
    label = []
    for i in range(len(M)):
        label.append(np.argmax(M[i]))
    return label

## Stop the refresh if Matrix change a little (<1e-7)
def Stop_condition(M_new, M_old):
    global Epsilon
    sign = 0
    for i in range(len(M_new)):
        for j in range(len(M_new[0])):
            if abs(M_new[i][j] - M_old[i][j]) < Epsilon :
                sign += 1
            if sign >= 120 : return True  
    return False

## Initialize Membership Matrix
def InitializeMembershipMatrix(samples, k):
    global MAX
    M = [];
    for i in range(len(samples)):
        current = []
        rand_sum = 0.0
        for j in range(k):
            dummy = np.random.randint(1, int(MAX))
            current.append(dummy)
            rand_sum += dummy
        for m in range(k):
            current[m] = current[m] / rand_sum   ##normalization
        M.append(current)
    return M

## normalize the MembershipMatrix
def NormalizeMembership(M):
    for i in range(len(M)):
        maximum = np.max(M[i])
        for j in range(len(M[0])):
            if M[i][j] == maximum:
                M[i][j] = 1
            else: M[i][j] = 0   ##取每个样本的最大归属值赋值为1,其余为0
    return M

## visualization module
def clusters_show(clusters,centers, step):
    color = ["red", "blue", "green","black"]
    marker = ["*", "^", "D","X"]
    label  = ["Category I", "Category II", "Category III", "Category IV"]
    plt.figure(figsize=(8, 8))
    plt.title("step: {}".format(step))
    plt.xlabel("Density", loc="center")
    plt.ylabel("Sugar Content", loc="center")

    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        for j in range(len(cluster)):
            plt.scatter(cluster[j][0], cluster[j][1], c=color[i],marker=marker[i],s=80)

    for i in range(len(clusters)):
        plt.scatter(centers[0][i][0],centers[0][i][1],marker="+", c=color[i])
    plt.show()

## computing cluster center computing module
def compute_center(samples,k,b):
    step = 0
    Or_centers = []
    Or_M = []
    M = InitializeMembershipMatrix(samples, k)
    while True:
        M_old = M.copy()
        centers = []
        label = []
        clusters_center = []
        for i in range(k):
            dummy_sum_num = 0.0
            dummy_sum_dum = 0.0
            for j in range(len(samples)):
                dummy_sum_num += (M[j][i] ** b) * samples[j]
                dummy_sum_dum += (M[j][i] ** b)
            clusters_center.append(dummy_sum_num / dummy_sum_dum)
        centers.append(clusters_center)
        if step == 0:
            Or_centers = centers
            Or_M = M

        # calculate distance matrix
        distance_matrix = []
        for i in range(len(samples)):
            current = []
            for j in range(k):
                current.append(distance(samples[i], centers[0][j]))
            distance_matrix.append(current)

        clusters = [[] for i in range(k)]
        label = Distance(M)
        for i in range(len(samples)):
            clusters[label[i]].append(samples[i])

        clusters_show(clusters,centers, step)
        # Refresh MembershipMatrix
        M = []
        for i in range(k):
            for j in range(len(samples)):
                dummy = 0.0
                dummy_num = 0.0
                for m in range(k):
                    dummy += (1 / (distance_matrix[j][m])**2) ** (1 / (b-1))
                dummy_num = (1 / (distance_matrix[j][i])**2) ** (1 / (b-1))
                M.append(dummy_num / dummy)
        M = np.array(M).reshape(4,30).T
        # Stop condition Judge (Change of cluster center lags behind the degree of attribution)
        if Stop_condition(M,M_old):
            break
        step += 1 
        
        #print("step:{}".format(step), "\n", "centers:{}".format(centers))
    return Or_centers,Or_M, centers, step, M,

if __name__ == '__main__':

    make_seed()
    k = 4 ##Cluster number
    b = 2.5 ##Fuzzy parameters
    labels = []
    samples=np.array([
    [0.697,0.460],[0.774,0.376],[0.634,0.264],[0.608,0.318],[0.556,0.215],
    [0.403,0.237],[0.481,0.149],[0.437,0.211],[0.666,0.091],[0.243,0.267],
    [0.245,0.057],[0.343,0.099],[0.639,0.161],[0.657,0.198],[0.360,0.370],
    [0.593,0.042],[0.719,0.103],[0.359,0.188],[0.339,0.241],[0.282,0.257],
    [0.748,0.232],[0.714,0.346],[0.483,0.312],[0.478,0.437],[0.525,0.369],
    [0.751,0.489],[0.532,0.472],[0.473,0.376],[0.725,0.445],[0.446,0.459]])
    Or_centers,Or_M, centers, step, MembershipMatrix = compute_center(samples,k,b)
    for j in range(len(MembershipMatrix)):
            labels.append(np.argmax(MembershipMatrix[j])+1)
    #print("初始聚类中心为:",np.array(Or_centers),sep="\n")
    print("初始隶属度为:",np.array(Or_M),sep="\n")
    print("分类完成!共迭代%d次"%(step))
    print("共分为%d类,最终分类的结果为:"%(k))
   # print("优化后的隶属度矩阵为:",MembershipMatrix,sep="\n")
    #print(MembershipMatrix)
    for i in range(k):
        label = []
        print("第 %d 类最终聚类中心为:"%(i+1),centers[0][i])
        label.append(np.where(np.array(labels) == (i+1)))
        print("类中数据的编号为:",np.array(label)+1)
        
        

