
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import tensorflow as tf

def KernelHyperParameterLearning(iteration, learningRate, trainingX, trainingY):
    numDataPoints = len(trainingY)
    numDimension = len(trainingX[0])
 
    # Input and Output Declaration for tensorflow
    obsX = tf.placeholder(tf.float32, [numDataPoints,numdimension])
    obsY = tf.placeholder(tf.float32, [numDataPoints, 1])
 
    # Learning Parameter Variable Declaration for tensorflow
    theta0 = tf.Variable(1.0)
    theta1 = tf.Variable(1.0)
    theta2 = tf.Variable(1.0)
    theta3 = tf.Variable(1.0)
    beta = tf.Variable(10.0)
 
    # Kernel building
    matCovarianceLinear = []
    for i in range(numDataPoints):
        for j in range(numDataPoints):
            kernelEvaluationResult =kernelFunctionWithTensorflow(theta0, theta1,theta2, theta3, 
                                                                 tf.slice(obsX, [i,0], [1,numdimension]), 
                                                                 tf.slice(obsX, [j,0], [1,numDimension]))
            if i != j:
                matCovarianceLinear.append(kernelEvaluationResult)
            if i == j:
                matCovarianceLinear.append(kernelEvaluationResult+tf.div(1.0,beta))
    matCovarianceCombined =tf.pack(matCovarianceLinear)
    matCovariance =tf.reshape(matCovarianceCombined, [numDataPoints, numDataPoints])
    matCovarianceInv =tf.inv(matCovariance)

def KernelFunctionWithTensorflow(theta0, theta1, theta2, theta3, X1,X2):
    insideexp1 = tf.mul(tf.div(theta1, 2.0), np.dot((X1-X2), (X1-X2)))
    insideexp2 = theta2
    insideexp3 = tf.mul(theta3, np.dot(np.transpose(X1), X2))
    insideexp = tf.add(tf.add(insideexp1, insideexp2),insideexp3)
    ret = tf.mul(theta0, tf.exp(insideexp))
    return ret


# In[ ]:




