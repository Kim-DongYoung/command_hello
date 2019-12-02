
# coding: utf-8

# In[18]:


import numpy as np

# 퍼셉트론을 구현하는 클래스를 정의
class perceptron():
    
    # 퍼셉트론 클래스의 생성자를 정의(디폴트값)
    # 임계값, learning rate, 학습 횟수
    def __init__(self, thresholds=0.0, eta=0.01, n_iter=10):
        self.thresholds = thresholds
        self.eta = eta
        self.n_iter = n_iter
    
    # 트레이닝 데이터 X와 실제 결과값 y를 인자로 받아 머신러닝을 수행하는 함수
    def fit(self, X, y):
        
        # 가중치를 numpy 배열로 정의
        # X.shape[1] = 트레이닝 데이터의 입력값 개수
        # X.shape 값이 (4,2)일 경우 X.shape[1] 값은 2가 되며 np.zeros(3)이 된다.
        # 이 때 numpy 배열은 [0.0.0.]이 된다.
        self.w_ = np.zeros(1+X.shape[1])
        
        # 머신러닝 반복 횟수에 따라 퍼셉트론 예측값의 실제 결과값이 다른 오류 횟수를
        # 저장하기 위한 변수
        self.errors_ = []
        
        # self.n_iter로 지정한 숫자만큼 반복함.
        # 자세한 내용은 http://blog.naver.com/samsjang/220955881668 참조
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target-self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update!=0.0)
            self.errors_.append(errors)
            print(self.w_)
            
        return self
    
    # np.dot(x,y) : 행렬곱(내적)
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    # 순입력 함수 결과값이 임계값인 self.thresholds의 값보다 크면 1, 그렇지 않으면 -1을 
    # 리턴하는 코드
    def predict(self, X):
        return np.where(self.net_input(X) > self.thresholds, 1, -1)
    


# In[ ]:




