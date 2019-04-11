# -*- coding:utf8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt


def save_result(points1, points2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x1 = [idx for idx in range(len(points1))]
    y1 = points1
    y2 = points2
    l1 = plt.plot(x1, y1, 'r--', label='predictions')
    l2 = plt.plot(x1, y2, 'g--', label='GT')
    plt.plot(x1, y1, 'ro-', x1, y2, 'g+-')
    plt.title('predictions VS GT')
    plt.legend()
    plt.savefig('./image/testdata.png')


x= np.random.uniform(0,50,1000)
x1= np.random.uniform(400,700,1000)
x2= np.random.uniform(-20,20,1000)
x3= np.random.uniform(0.1,6.8,1000)
y=1.85*x + 1.7*x1 + 0.5*x2 + 1.3*x3+ np.random.normal(-3,5,1000)

c=np.vstack((x,x1,x2,x3,y)) #合并
c=c.swapaxes(1,0) #交换行列
np.savetxt('./test.csv', c, delimiter=' ')

# save_result(c[0], infer_label)

# 多维的不适用
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('./testdata.png')
plt.show()