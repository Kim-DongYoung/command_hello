
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from matplotlib.colors import ListedColormap


# In[1]:

krfont = {'family':'HYGothic-Medium', 'weight':'bold', 'size':10}
matplotlib.rc('font', **krfont)
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_decision_region(X, y, classifier, test_idx=None, resolution=0.02, title=''):
    # markers와 colors의 튜플을 선언함.
    # 튜플은 리스트와 비슷하지만 수정이 불가한 리스트
    markers = ('s', 'x', 'o','^', 'v')
    colors = ('r', 'b', 'lightgreen', 'gray', 'cyan')
    # np.unique(y)는 y에 있는 고유한 값을 작은 값 순으로 나열
    # 예를 들어 [1,0,0,2,2,0,0,1,2] 에는 고유한 값이 0,1,2밖에
    # 없으므로 [0,1,2] 가 된다.
    # 따라서 cmap = ListedColormap(colors[:3])이 되어 cmap에는 colors[0],
    # colors[1], colors[2]가 매핑된 ListedColormap 객체가 된다.
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # decision surface 그리기
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], c=cmap(idx), marker=markers[idx], label=cl)
        
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', linewidth=1, marker='o', s=80, label='테스트셋')
        
    plt.xlabel('표준화 된 꽃잎 길이')
    plt.ylabel('표준화 된 꽃잎 너비')
    plt.legend(loc=2)
    plt.title(title)
    plt.show()   


# In[ ]:




# In[ ]:



