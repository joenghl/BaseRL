import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
plt.style.use('seaborn-whitegrid')
palette = pyplot.get_cmap('Set1')
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}

fig=plt.figure(figsize=(20,10))
iters=list(range(7))
#这里随机给了alldata1和alldata2数据用于测试
alldata1=[]#算法1所有纵坐标数据
data=np.array([2,4,5,8,11,13,15])#单个数据
alldata1.append(data)
data=np.array([2,3,6,12,13,13,15])
alldata1.append(data)
data=np.array([2,2,7,9,13,14,16])
alldata1.append(data)
alldata1=np.array(alldata1)
alldata2=[]#算法2所有纵坐标数据
data=np.array([2,4,5,8,10,10,11])#单个数据
alldata2.append(data)
data=np.array([3,3,3,6,7,8,10])
alldata2.append(data)
data=np.array([3,3,5,5,6,7,9])
alldata2.append(data)
alldata2=np.array(alldata2)

for i in range(2):
    color=palette(0)#算法1颜色
    ax=fig.add_subplot(1,2,i+1)   
    avg=np.mean(alldata1,axis=0)
    std=np.std(alldata1,axis=0)
    r1 = list(map(lambda x: x[0]-x[1], zip(avg, std)))#上方差
    r2 = list(map(lambda x: x[0]+x[1], zip(avg, std)))#下方差
    ax.plot(iters, avg, color=color,label="algo1",linewidth=3.0)
    ax.fill_between(iters, r1, r2, color=color, alpha=0.2)
    
    color=palette(1)
    avg=np.mean(alldata2,axis=0)
    std=np.std(alldata2,axis=0)
    r1 = list(map(lambda x: x[0]-x[1], zip(avg, std)))
    r2 = list(map(lambda x: x[0]+x[1], zip(avg, std)))
    ax.plot(iters, avg, color=color,label="algo2",linewidth=3.0)
    ax.fill_between(iters, r1, r2, color=color, alpha=0.2)
    
    ax.legend(loc='lower right',prop=font1)
    ax.set_xlabel('Outer loop iterations',fontsize=22)
    ax.set_ylabel('Objectives',fontsize=22)
plt.show()
