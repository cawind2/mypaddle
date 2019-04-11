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


x= np.random.uniform(0,50,600)
y=1.85*x + np.random.normal(0,3,600)

c=np.vstack((x,y)) #合并
c=c.swapaxes(1,0) #交换
np.savetxt('./test.csv', c, delimiter=' ')

# save_result(c[0], infer_label)

plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('./testdata.png')
plt.show()