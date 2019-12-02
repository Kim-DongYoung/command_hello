
# coding: utf-8

# In[3]:


import numpy as np
from numpy.random import seed

# 아달라인 - 확률적 경사하강법
# adaline - Stochastic Gradient Descent
class AdalineSGD():
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        
        # random_state의 값이 있으면 이 값으로 난수 발생기를 초기화
        if random_state:
            seed(random_state)
        
    def fit(self, X, y):
        self.w_ = np.zeros(1+X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            
            # self.shuffle이 True로 설정되어 있으면, self._shuffle() 함수를 이용해
            # 트레이닝 데이터 X와 y를 랜덤하게 섞는다.
            if self.shuffle:
                X, y = self._shuffle(X, y)
            
            cost = []
            for xi, target in zip(X, y):
                output = self.net_input(xi)
                error = target - output
                self.w_[1:] += self.eta * xi.dot(error)
                self.w_[0] += self.eta * error
                cost.append(0.5 * error**2)
                
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
            
        return self
    
    # numpy.random.permutation은 주어진 인자 미만의 정수(0을 포함)으로 순열을
    # 만드는 함수.
    # r의 값은 0~len(y) 미만까지 정수를 랜덤하게 섞은 결과임.
    # 따라서 X[r], y[r]은 X와 y를 랜덤하게 섞은 numpy 배열이 된다.
    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:] + self.w_[0])
    
    def predict (self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# In[ ]:




