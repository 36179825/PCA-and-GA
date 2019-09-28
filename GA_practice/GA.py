# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:00:32 2017

@author: Jasmine
"""

import numpy as np
import random
import math

pop_size = 20 #做20個
pop_len = 5
crover_rate = 0.5 #有0.5的機率會做crossover
mu_rate = 0.05 #有0.05的機率會做mutation
iteration = 1
x_sum = 0
maxi = 0
x_bin = np.zeros((pop_size, pop_len))
s_bin = np.zeros((pop_size, pop_len))
c_bin = np.zeros((2, pop_len))
x_square = np.zeros((pop_size,1))
fs_range = np.zeros((pop_size,2))

for i in range (pop_size): #random出20個且長度為5的0,1矩陣
    x_bin[i,:] = np.random.choice([0,1], pop_len) 
#==============================================================================
# 將x_bin的內容給x_bin_origin, 
# 目的是等一下在做疊代訓練時會改變x_bin的內容, 
# x_bin_origin是保留原本random的0,1矩陣
#==============================================================================
#x_bin_origin = x_bin
x_ten = np.zeros((pop_size,1)) #做二進制轉十進制存放的空間
x = np.zeros((pop_size,1))      #公式會用到的
fs = np.zeros((pop_size,1))

best_fs = -1     
best_fs_i = 0   
worst_fs = 33*2
worst_fs_i = 0 

#下面的所有步驟做20次
for i in range (iteration):
    x_ten = np.zeros((pop_size,1))
    for fre in range (pop_size): 
         #binary to decimal
         for dec in range (pop_len):
             x_ten[fre,0] = x_ten[fre,0] + x_bin[fre,dec] * (2**dec)
         #算fitness
         x_square[fre,0] = x_ten[fre,0]**2 
#          if(best_fs < x_square[fre,0]): #假如best_fs小於f(i)
#              # index of best fs
#              best_fs_i = fre      #將fre這個index指給best_fs_i做紀錄，表示第幾個為最好的答案
#              # best fs value
#              best_fs = x_square[fre,0] # 將f(i)指給best_fs
#          # worst fs
#          if(worst_fs > x_square[fre,0]): #假如最壞的worst_fs這個值大於f(i)這個值
#          # index of worst fs
#              worst_fs_i = fre;     # 將index i 指給worst_fs_i
#          # worst fs value
#              worst_fs = x_square[fre,0];
#  
    #normal x_square
    x_sum = np.sum(x_square)
    #for normal in range (pop_size):
    fs = x_square/x_sum
    #print(np.sum(fs))

    #create the range of fitness
    for fre in range (pop_size):
        if fre==0:
            #fre[fre,0]
            fs_range[fre,0] = 0
            fs_range[fre,1] = fs[fre,0]
        elif fre== pop_size-1:
            fs_range[fre,0] = fs_range[fre-1,1]
            fs_range[fre,1] = 1
        else:
            for r in range (1, pop_size-1):
                fs_range[r,0] = fs_range[r-1,1]
                fs_range[r,1] = fs_range[r-1,1]+ fs[r,0]
#==============================================================================
#     
#     # generate new population
#     ranchange_num = random.randrange(2, 20, 2) 
#     x_change = np.zeros((ranchange_num, pop_len))
#     x_change_num = np.zeros((ranchange_num, 1))
#     for gen in range (ranchange_num):
#         random_num = np.float64(random.randint(0,100)/100)
#         for ran in range (pop_size):# for 1~20個
#             if random_num >= fs_range[ran,0] and random_num <= fs_range[ran,1]:
#                 x_change[gen,:] = x_bin[ran,:]
#                 x_change_num[gen,0] = ran
#     
#==============================================================================
    # generate new population
    for ran in range (pop_size):# for 1~20個
        random_num = np.float64(random.randint(0,100)/100)
        for rann in range (pop_size):
            if random_num >= fs_range[rann,0] and random_num <= fs_range[rann,1]:
                s_bin[rann,:] = x_bin[rann,:]

    #crossover
    for cro in range (0,pop_size,2):
        random_num_cro = np.float64(random.randint(0,100)/100)
        if random_num_cro >= crover_rate:
            random_cut = random.randint(1,pop_len-1)
            c_bin[0,:] = s_bin[cro,:]
            c_bin[1,:] = s_bin[cro+1,:]
            for n in range (random_cut,pop_len):
                s_bin[cro,n]= c_bin[1,n]
                s_bin[cro+1,n]= c_bin[0,n]

    #mutation
    for mu in range (pop_size):
        for zeroone in range (pop_len):
            random_num_cro = np.float64(random.randint(0,100)/100)
            if random_num_cro <= mu_rate:
                if s_bin[mu,zeroone] == 0:
                    s_bin[mu,zeroone] = 1
                else: s_bin[mu,zeroone] = 0
    x_ten = np.zeros((pop_size,1))
    x_square = np.zeros((pop_size,1))
    for fre in range (pop_size):
         for dec in range (pop_len):
             x_ten[fre] = x_ten[fre] + s_bin[fre,dec] * (2**dec)
             x_square[fre] = x_ten[fre]**2 
    #x_ten_temp = np.zeros((pop_size,1))

            
#==============================================================================
#     #update x_bin
#     
#     for upd in range (ranchange_num):
#         a = int(x_change_num[upd,0])
#         if x_ten[upd,0] < x_ten_temp[upd,0]:
#             x_bin[a,:] = x_change[upd,:]
#==============================================================================

    #choose the maxima to  be the answer
    for ma in range (pop_size):
        if maxi < x_square[ma]:
            maxi = x_square[ma]

print(math.sqrt(maxi))             
            