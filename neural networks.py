# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 13:51:04 2018

@author: ANUBHAV SHUKLA
"""

import numpy as np

#set seed value 
np.random.seed(0)
# intializing parameters

#Setting up learning rate, weights and bias values

# first laye is x1 2nd layer is x2 and 3rd year is output
#sample input (insert your training data set in given manner)
Input=np.array([[0.09,0.10,0.14,0.18],[0.04,0.10,0.21,0.24],[1.0, 1.0, 1.0, 1.0]])
n= 0.5                          #learning rate
W=np.random.random((2, 3))      #weight of layer 1
V=np.random.random((3, 3))      #weight of layer 2
U=np.random.random((3, 1))      #weight of layer 3
Bw=np.zeros((1, 3))             #bias value of layer 1 (initially zero)
Bv=np.zeros((1, 3))             #bias value of layer 2 (initially zero)
Bu=np.zeros((1,1))              #bias value of layer 3 (initially zero)

#defining tan-sigmoid and sigmoid
def sigmoid(x):
    return (1/(1+np.exp(-x)))
def sigm_derivative(x):
    return (x*(1-x))


# number of iterations to train each point 
iter= 20

#iteration
for j in range (0,4):
    X=np.array([[Input[0][j],Input[1][j]]])
    T=Input[2][j]
    
    for i in range (0,iter):
        #output from hidden layers
        F2=X.dot(W)+Bw
        O2=sigmoid(F2)
        
        F3=O2.dot(V)+Bv
        O3=sigmoid(F3)
        
        #final output
        F4=O3.dot(U)+Bu
        Y=sigmoid(F4)
        
        #Loss function
        E=0.5*(T-Y)**2
              
        #adjusting weights and bais value through backpropagation
        
        #Slopes
        S4=sigm_derivative(Y)
        S3=sigm_derivative(O3)
        S2=sigm_derivative(O2)
        
        #Error associated with each layer and delta
        D4=E*S4
        E3=D4.dot(np.transpose(U))
        D3=E3*S3
        E2=D3.dot(np.transpose(V))
        D2=E3*S2
        
        #updating weights and bias value
        U= U+ np.transpose(O3).dot(D4)*n
        Bu= Bu+ D4*n
        V= V+ np.transpose(O2).dot(D3)*n
        Bv= Bv+ D3*n
        W= W+ np.transpose(X).dot(D2)*n
        Bw= Bw+ D2*n

#sample value to check (change to check another value)
test_X=np.array([[0.22,0.28]])                      
F2=X.dot(W)+Bw
O2=sigmoid(F2)        
F3=O2.dot(V)+Bv
O3=sigmoid(F3)
F4=O3.dot(U)+Bu
Y=sigmoid(F4)

print("Calculated output value is ",Y)
