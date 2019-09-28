# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:00:32 2017

@author: Jasmine
"""

import numpy as np
import random
import math

pop_size = 50 #做50個
pop_len = 22
crover_rate = 0.25 #有0.25的機率會做crossover
mu_rate = 0.01 #有0.01的機率會做mutation
iteration = 150 #做150次iteration
x_sum = 0
maxi = 0
x_bin = np.zeros((pop_size, pop_len)) #x_bin: size為(50, 22) 的0矩陣
s_bin = np.zeros((pop_size, pop_len)) #s_bin: size為(50, 22) 的0矩陣
c_bin = np.zeros((2, pop_len)) #c_bin: size為(2, 22) 的0矩陣
x = np.zeros((pop_size,1)) #x: size為(50, 1) 的0矩陣
f = np.zeros((pop_size,1)) #f: size為(50, 22) 的0矩陣
fs_range = np.zeros((pop_size,2)) #fs_range: size為(50, 22) 的0矩陣

for i in range (pop_size): #random出50個且長度為22的0,1矩陣
    x_bin[i,:] = np.random.choice([0,1], pop_len) 
fs = np.zeros((pop_size,1)) #fs: size為(50, 22) 的0矩陣

best_fs = -1 #把最佳fitness 預設為 -1    
best_fs_i = 0 #把最佳x 設為 0

#下面的所有步驟做iteration次
for i in range (iteration): #for i in range (iteration): 表示變數i的值從0~iteration-1
    x_ten = np.zeros((pop_size,1)) #做二進制轉十進制存放的空間, 每一輪都要歸零
    for fre in range (pop_size): 
        #binary to decimal
        for dec in range (pop_len):
            x_ten[fre,0] = x_ten[fre,0] + x_bin[fre,dec] * (2**dec) #2**1 表示2的一次方, 依此類推
        #轉為-1~2之間的值
        x[fre,0] = x_ten[fre,0] * 3 / (2**pop_len - 1) - 1
        # caculate fitness
        f[fre,0]  = x[fre,0] * math.sin(10 * math.pi * x[fre,0]) + 1
        if(best_fs < f[fre,0]): #假如best_fs小於f(i)
            # index of best fs
            best_fs_i = x[fre,0] #將x[fre,0]這個index指給best_fs_i做紀錄, 表示最好的答案是哪個x值
            # best fs value
            best_fs = f[fre,0] #將最好的f[fre,0]存進best_fs         
            print("x :",best_fs_i,"fitness :",best_fs)

    #算機率
    x_sum = np.sum(f) #算f裡所有值加總
    fs = f/x_sum #每個值除以總合

    #create the range of fitness
    for fre in range (pop_size):
        if fre==0: #若fre = 0(第1個數)
            fs_range[fre,0] = 0  #fs_range用來存某個數值的機率範圍, row=0存起始的值, row = 1存結束的值
            fs_range[fre,1] = fs[fre,0]
        elif fre== pop_size-1: #若fre = pop_size-1(最後一個數)
            fs_range[fre,0] = fs_range[fre-1,1] 
            fs_range[fre,1] = 1
        else:
            for r in range (1, pop_size-1):
                fs_range[r,0] = fs_range[r-1,1]
                fs_range[r,1] = fs_range[r-1,1]+ fs[r,0]

    # generate new population
    for ran in range (pop_size): # for 1~20個
        random_num = np.float64(random.randint(0,100)/100) #隨機取一個0~1之間的值(用(random 0~100某數)/100)實現
        for rann in range (pop_len):
            if random_num >= fs_range[rann,0] and random_num <= fs_range[rann,1]: #若random_num落在範圍內, 就選那個數
                s_bin[ran,:] = x_bin[rann,:] #把x_bin rann column的所有值存進s_bin的ran column中
    #crossover
    for cro in range (0,pop_size,2): #for cro in range (0,pop_size,2): 表示變數cro的值從0~pop_size-1,cro每次+2
        random_num_cro = np.float64(random.randint(0,100)/100) #隨機取一個0~1之間的值(用(random 0~100某數)/100)實現
        if random_num_cro <= crover_rate:
            random_cut = random.randint(1,pop_len-1) #random要交換的位置
            c_bin[0,:] = s_bin[cro,:] #將要做crossover的兩個的原本值存在c_bin
            c_bin[1,:] = s_bin[cro+1,:]
            for n in range (random_cut,pop_len): #做交換
                s_bin[cro,n]= c_bin[1,n]
                s_bin[cro+1,n]= c_bin[0,n]

    #mutation
    for mu in range (pop_size):
        for zeroone in range (pop_len):
            random_num_cro = np.float64(random.randint(0,100)/100) #隨機取一個0~1之間的值(用(random 0~100某數)/100)實現
            if random_num_cro <= mu_rate:
                if s_bin[mu,zeroone] == 0: #原本若為0則改為1
                    s_bin[mu,zeroone] = 1
                else: s_bin[mu,zeroone] = 0
    x_bin = s_bin #把做完所有步驟的值存回x_bin, 以用新的x_bin做下一輪運算