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
Input=np.array([[0.05,0.09,0.12,0.15],[0.02,0.11,0.20,0.22],[0.50474,0.51224,0.51899,0.52223]])
n= 0.08
W=np.random.random((2, 3))
V=np.random.random((3, 3))
U=np.random.random((3, 1))
Bw=np.zeros((1, 3))
Bv=np.zeros((1, 3))
Bu=np.zeros((1,1))

#defining tan-sigmoid and sigmoid
def sigmoid(x):
    return (1/(1+np.exp(-x)))
def sigm_derivative(x):
    return (x*(1-x))


# number of iterations 
iter= 100

#iteration
for j in range (0,4):
    
    X=np.array([[Input[0][j],Input[1][j]]])
    T=Input[2][j]
    for i in range (0,10):
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
        Bu= Bu+ np.sum(D4, axis=0)*n
        V= V+ np.transpose(O2).dot(D3)*n
        Bv= Bv+ np.sum(D3, axis=0)*n
        W= W+ np.transpose(X).dot(D2)*n
        Bw= Bw+ np.sum(D2, axis=0)*n
