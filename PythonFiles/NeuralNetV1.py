import plotly.plotly as py
import plotly.figure_factory as ff
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np 
import math

class Matrix:
    def __init__(self, rows,cols,vector):
        self.activation = "sigmoid"
        if vector is not None:
            self.rows = 1
            self.cols = len(vector)
            self.mat = [[random.random() for col in range(self.cols)] for row in range(self.rows)]
            for i in range(self.cols):
                self.mat[0][i] = vector[i]
            return
        self.rows = rows
        self.cols = cols    
        self.mat = [[random.random() for col in range(cols)] for row in range(rows)]
        for i in range(rows):
            for j in range(cols):
                self.mat[i][j] = random.random()
                #self.mat[i][j] = 0.5
        #print (self.mat)

    def setCol(self, X ,col): 
        for i in range(self.rows):
            self.mat[i][col] = X[i] 
                
    def colView(self, c):
        newmat =  Matrix(1,self.rows,None)
        for i in range(self.rows):
            newmat.mat[0][i] = self.mat[i][c]
        return newmat
    
    def add(self, X,Y):
        for i in range(X.rows):
            for j in range(Y.cols):
                self.mat[i][j] = X.mat[i][j] + Y.mat[i][j]
         
    def mapActivation(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.activation == "sigmoid" :
                    x = self.mat[i][j]
                    self.mat[i][j] = 1 / (1 + np.exp(-x))
                else:
                    self.mat[i][j] = np.tanh(self.mat[i][j])
     
    def mapDeActivation(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.activation == "sigmoid" :
                    y = self.mat[i][j]
                    self.mat[i][j] =  y * (1 - y)
                else:
                    y = self.mat[i][j]
                    self.mat[i][j] = (1.0 - (y*y))

                
    def subtract(self, X,Y):
        for i in range(X.rows):
            for j in range(X.cols):
                self.mat[i][j] = (X.mat[i][j]) - (Y.mat[i][j])
    def copy(self, X):
        for i in range(X.rows):
            for j in range(X.cols):
                self.mat[i][j] = X.mat[i][j]  
                
    def copyTranspose(self, X):
        for i in range(X.rows):
            for j in range(X.cols):
                self.mat[j][i] = X.mat[i][j]  
                
    def multiplyScalar(self,X, s):
        #print (self.rows,self.cols,X.rows,X.cols,Y.rows,Y.cols)
        for i in range(X.rows):
            for j in range(X.cols):
                    self.mat[i][j] = (self.mat[i][j])*(X.mat[i][j])*(s)
                    
    def multiply(self, X,Y):
        if X.cols != Y.rows:
            print("ERROR in Multiplication")
        for i in range(X.rows):
            for j in range(Y.cols):
                self.mat[i][j] = 0
                for k in range(Y.rows):
                    self.mat[i][j] = self.mat[i][j] + (X.mat[i][k] * Y.mat[k][j])

    def multiplyTranspose(self, X,Y):
        if X.cols != Y.cols:
            print("ERROR in MultiplicationTranspose")
        for i in range(X.rows):
            for j in range(Y.rows):
                self.mat[i][j] = 0
                for k in range(Y.cols):
                    self.mat[i][j] = self.mat[i][j] + (X.mat[i][k] * Y.mat[j][k])
                    
class Layer:
    def __init__(self, curr_lay,prev_lay):     
        self.nodes = curr_lay
        self.weights = Matrix(curr_lay,prev_lay,None)
        
        #initialise the weights accordingly
        self.weights.mat = np.random.randn(curr_lay,prev_lay)*np.sqrt(2/prev_lay)
        #print("New weights :",curr_lay," :",self.weights.mat)
        
        self.weights_T = Matrix(prev_lay, curr_lay,None)
        self.weights_delta = Matrix(curr_lay,prev_lay,None)
        self.error_val =  Matrix(curr_lay,1,None) 
        self.output =  Matrix(curr_lay,1,None) 
        self.output_T =  Matrix(1, curr_lay,None)
        self.bias =  Matrix(curr_lay,1,None) 
        self.gradients = Matrix(curr_lay,1,None) 
        
    
class NeuralNet:
    def __init__(self, layers):
        self.layercount = len(layers)
        self.layers = []
        self.learning_rate = 0.08
        self.errorval = 0
        self.max_batch_size = 1
        self.debug = False
        self.data_index = 1
        
        self.cumulative_error =0 # sum of error so far
        self.data_count=0
        self.error_percentage=100
        
        prev_size = 1
        for curr_size in layers:
            self.layers.append(Layer(curr_size,prev_size))
            prev_size = curr_size
        print ("NeuralNet created:  layers:",self.layercount)
        
        
    def predict(self, input_args, target_args, batch_size):
        batch_size = 1
        target_mat = Matrix(0,0,target_args)
        self.layers[0].output.setCol(input_args,0)
            
        i = 1
        while i < (self.layercount):   
            self.layers[i].output.multiply(self.layers[i].weights, self.layers[i-1].output)
            self.layers[i].output.add(self.layers[i].output,self.layers[i].bias)
            self.layers[i].output.mapActivation()
            i = i+1
                
        last_layer = self.layercount-1
        self.layers[last_layer].error_val.subtract(target_mat, self.layers[last_layer].output)
        self.errorval = self.layers[last_layer].error_val.mat[0][0]
        
        # Stats : calculate the error
        self.cumulative_error = self.cumulative_error + abs(self.layers[last_layer].output.mat[0][0]-target_mat.mat[0][0])
        self.data_count = self.data_count +1
        self.error_percentage = (self.cumulative_error/self.data_count)*100
       
        
    def train(self, input_args, target_args, batch_size):
        self.predict(input_args,target_args,batch_size)

        if self.debug==True :
            #print("weights:",self.layers[1].weights.mat)
            print(self.data_index," Input:", input_args, "Target:", target_args, " : ",self.layers[self.layercount-1].output.mat[0][0])
            
        i = self.layercount-1
        while i>0 :
            self.layers[i].gradients.copy(self.layers[i].output)
            self.layers[i].gradients.mapDeActivation()
            self.layers[i].gradients.multiplyScalar(self.layers[i].error_val,self.learning_rate)
            
            #self.layers[i-1].output_T.copyTranspose(self.layers[i-1].output)
            #self.layers[i].weights_delta.multiply(self.layers[i].gradients, self.layers[i-1].output_T)
            self.layers[i].weights_delta.multiplyTranspose(self.layers[i].gradients, self.layers[i-1].output)
    
            self.layers[i].weights.add(self.layers[i].weights, self.layers[i].weights_delta)
            self.layers[i].bias.add(self.layers[i].bias, self.layers[i].gradients)
            if (i-1)>0 :
                self.layers[i].weights_T.copyTranspose(self.layers[i].weights)
                self.layers[i-1].error_val.multiply(self.layers[i].weights_T, self.layers[i].error_val) 
            i = i-1
            


