# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:16:35 2017

@author: Jasmine
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from skimage import io

trainNum = 39
w,h = 100,120
x_train = np.zeros((trainNum*2, w*h))
x_train_ = np.zeros((w*h, trainNum*2))
energy = 0.85

#load Data

for i in range(trainNum):
    x_train[i, :] = np.array(Image.open('D:\\DNN\\DB\\Man(50)\\Training\\'+str(i+1)+'.jpg'),dtype=np.float).reshape(1, -1)
for i in range(trainNum):
    x_train[i+39, :] = np.array(Image.open('D:\\DNN\\DB\\Woman(50)\\Training\\'+'W('+str(i+1)+')'+'.jpg'),dtype=np.float).reshape(1, -1)

#x_train normalize

x_train = x_train/255.

#x_train(78, 12000)

x_train_t = x_train.T #(12000, 78)
imgsum = x_train.shape[0]
#mean
x_mean = np.mean(x_train, axis=0) #(12000, )
 

#x_train_t - x_mean
x_train_ = np.matrix(x_train_t) - np.matrix(x_mean).T #(12000, 78) - (12000, 1) = (12000, 78)

# C = covariance matrix
C = np.matrix(x_train_).T * np.matrix(x_train_)
C /= imgsum #(78, 78)

#eigenvalue, eigenvector                         
egvalue, egvector = np.linalg.eig(C)
#egvalue (78, ) 
#egvector (78, 78)

#sort
sort_indices = egvalue.argsort()[::-1]                             
egvalue = egvalue[sort_indices] #(78,)                           
egvector = egvector[sort_indices] #(78, 78)

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
egvector = egvector[0:evalues_count] #(78, evalues_count)

#
egvector = egvector.T
egvector = x_train_ * egvector
egvaluenorms = np.linalg.norm(egvector, axis=0)
egvector = egvector / egvaluenorms #(12000, 20)

#y
x_pca = np.matrix(egvector).T * np.matrix(x_train_) #(20, 78)
x_pca = np.array(x_pca)

#撒點
plt.scatter(x_pca[0, :39], x_pca[1, :39], c= "b" ,label= "male")
plt.scatter(x_pca[0, 39:78], x_pca[1, 39:78], c= "y" ,label= "female")
plt.legend(loc="lower left")

x = np.matrix(x_pca).T * np.matrix(egvector).T
x = np.matrix(x) + np.matrix(x_mean)

for i in range(5):
    for j in range(w*h):
        if x[i, j] <0:
            x[i, j] = 0
        elif x[i, j] >1:
            x[i, j] = 1
    y = x[i, :].reshape(h,w)
    y = y*255.
    plt.figure()
    io.imshow(y)
    