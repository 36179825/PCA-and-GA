# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 16:33:19 2017

@author: Jasmine
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

trainNum = 39
testNum = 11
w,h = 100,120
x_train = np.zeros((trainNum*2, h*w))
y_test = np.zeros((testNum*2, h*w))
x_train_ = np.zeros((h*w, trainNum*2))
energy = 0.85

#load Data

for i in range(trainNum):
    x_train[i, :] = np.array(Image.open('D:\\DNN\\DB\\Man(50)\\Training\\'+str(i+1)+'.jpg'),dtype=np.float).reshape(1, -1)
for i in range(trainNum):
    x_train[i+39, :] = np.array(Image.open('D:\\DNN\\DB\\Woman(50)\\Training\\'+'W('+str(i+1)+')'+'.jpg'),dtype=np.float).reshape(1, -1)

#x_train normalize
x_train = x_train/255.

#x_train(78, 12000)

#sum of images
imgsum = x_train.shape[0]

#mean
x_mean = np.mean(x_train.T, axis=1) #(12000, )

#x_train-mean
x_train_ = np.matrix(x_train) - np.matrix(x_mean) #(78, 12000)

# C = covariance matrix
C = np.matrix(x_train_).T * np.matrix(x_train_)
C /= imgsum #(12000, 12000)

##eigenvalue, eigenvector                         
egvalue, egvector = np.linalg.eig(C)
#egvalue (12000, ) 
#egvector (12000, 12000)

#sort
sort_indices = egvalue.argsort()[::-1]                             
egvalue = egvalue[sort_indices] #(12000,)                           
egvector = egvector[sort_indices] #(12000, 12000)

#取前85%的eigenvalu, eigenvector的數量(evalues_count)
evalues_sum = sum(egvalue[:])                                      
evalues_count = 0                                                       
evalues_energy = 0.0
for evalue in egvalue:
    evalues_count += 1
    evalues_energy += evalue / evalues_sum
    
    if evalues_energy >= energy:
        break

egvalue = egvalue[0:evalues_count] #(evalues_count, )
egvector = egvector[0:evalues_count] #(12000, evalues_count)

#
egvaluenorms = np.linalg.norm(egvector, axis=0)
egvector = egvector / egvaluenorms #(20, 12000)

#y
x_pca = np.matrix(egvector) * np.matrix(x_train_) #(78, 78)
x_pca = np.array(x_pca) #(?, 78)

#撒點
plt.scatter(x_pca[0, :39], x_pca[1, :39], c= "b" ,label= "male")
plt.scatter(x_pca[0, 39:78], x_pca[1, 39:78], c= "y" ,label= "female")
plt.legend(loc="lower left")

for i in range(testNum):
    y_test[i, :] = np.array(Image.open('D:\\DB\\Man(50)\\Testing\\'+str(i+40)+'.jpg'),dtype=np.float).reshape(1, -1)
    
for i in range(testNum):
    y_test[i+11, :] = np.array(Image.open('D:\\DB\\Woman(50)\\Testing\\'+'W('+str(i+40)+')'+'.jpg'),dtype=np.float).reshape(1, -1)

#y_test normalize
y_test = y_test/255.

#y_test-mean
y_test = np.matrix(y_test) - np.matrix(x_mean) 

#make label
y_label = np.matrix(y_test) * np.matrix(egvector.T) #(22, 12000) * (12000, 20) = (22, 20)

#
y_ = np.matrix(y_label) * np.matrix(egvector) #(22, 20) * (20, 12000) = (22, 12000)
y_ = np.matrix(y_) + np.matrix(x_mean)