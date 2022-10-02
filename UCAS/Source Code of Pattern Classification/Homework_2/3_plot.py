import numpy as np
import matplotlib.pyplot as plt
w = np.array([1, 0, 0, 1, 0, -1, 0, 0 ,0,
             2, 0, -2, -2, 0]).reshape(7,2)

#visibility
plt.figure()
for i in range(len(w)):
    if i < 3 :
        p1 = plt.scatter(w[i][0],w[i][1], color='red')
    else:
        p2 = plt.scatter(w[i][0],w[i][1], color='blue')
for j in range(4):
    plt.hlines(-1.5+j, xmin=-0.5, xmax=0.5)
    
#boundary
plt.vlines((-0.5, 0.5, -0.5, 0.5, 0.5),
        ymin=(-1.5, -0.5, 0.5, 1.5, -1.5),
        ymax=(-0.5, 0.5, 1.5, 2.5, -2.5))
plt.xlim(-2.5, 2)
plt.ylim(-2.5, 2.5)
plt.legend((p1,p2), ('w1','w2'), loc='upper left')
plt.show()