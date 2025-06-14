from Matrix import *
import pandas as pd
import random
import numpy as np 
                 
class Layer:
    def __init__(self, curr_lay,prev_lay,gpu_enabled,batch_size=1,last_layer=False):     
        self.nodes = curr_lay
        #activation = "leaky_relu"  
        activation = "sigmoid" 
        if last_layer :
            activation = "sigmoid"

        
        self.weights = Matrix(curr_lay,prev_lay,gpu_enabled, None, activation=activation)
        self.bias =  Matrix(curr_lay,1, gpu_enabled, None, activation=activation)  
        
        # weights_delta is a temporary matrix.
        self.weights_delta = Matrix(curr_lay,prev_lay, gpu_enabled, None, activation=activation)
        
        self.error_val =  Matrix(curr_lay,1,gpu_enabled , None, activation=activation) 
        self.output =  Matrix(curr_lay,1, gpu_enabled, None, activation=activation) 
        self.gradients = Matrix(curr_lay,1, gpu_enabled, None, activation=activation) 

        self.batch_error_val =  Matrix(curr_lay,batch_size,gpu_enabled , None, activation=activation)         
        self.batch_output =  Matrix(curr_lay,batch_size, gpu_enabled, None, activation=activation) 
        self.batch_gradients = Matrix(curr_lay,batch_size, gpu_enabled, None, activation=activation) 
        
        self.activation=activation
        
         
class NeuralNet:
    #__slots__ = ['layercount','layers','filename','learning_rate','batch_size','debug','data_index','load','cumulative_error','data_count','error_percentage','gpu_enabled','last_layer','batch_target','target_mat']
    def __init__(self, layers, gpu_enabled=False,name="default",load=False,batch_size=1,learning_rate=0.04):
        self.layercount = len(layers)
        self.layers = []
        self.learning_rate = learning_rate
        #self.errorval = 0
        self.batch_size = batch_size
        self.debug = False
        self.data_index = 1
        self.load = load
        
        self.cumulative_error=0 # sum of error so far
        self.data_count=0
        self.error_percentage=100
        self.gpu_enabled=gpu_enabled
        self.last_layer = self.layercount -1
        
        prev_size = 1
        current_layer=0
        last_layer = False
        for curr_size in layers:
            if current_layer == self.last_layer :
                last_layer = True
            current_layer = current_layer +1
            self.layers.append(Layer(curr_size,prev_size,gpu_enabled,batch_size=batch_size,last_layer=last_layer))
            prev_size = curr_size
        
        self.target_mat = Matrix(1,1,self.gpu_enabled, None)
        self.batch_target = Matrix(batch_size,1,self.gpu_enabled, None)
        print ("NeuralNetV4...  layers:",self.layercount, "GPU enabled:",gpu_enabled," batch size:",self.batch_size," activation: ",self.layers[0].activation," lastActivation: ",self.layers[self.last_layer].activation)
        i = 1
        filename  = name + "-" + str(self.layercount)
        while i < (self.layercount-1): 
            filename = filename + "-" + str(self.layers[i].nodes)
            i=i+1
        self.filename = filename  
        self.input_index =0
        if load:
            self.loadNeuralNet()     
        
    def saveNeuralNet(self):              
        print("Saving NeuralNet to filename : ",self.filename)
        wfile = self.filename + "-wgt-"
        bias = self.filename + "-bias-"
        i = 0
        while i < (self.layercount): 
            self.layers[i].weights.loadFromGpu()
            self.layers[i].bias.loadFromGpu()
            
            np.save(wfile+str(i),self.layers[i].weights.mat)
            #print(self.layers[i].weights.mat)
            np.save(bias+str(i),self.layers[i].bias.mat)         
            i=i+1
            
    def loadNeuralNet(self):              
        print("Loading NeuralNet from filename : ",self.filename)
        wfile = self.filename + "-wgt-"
        bias = self.filename + "-bias-"
        i = 0
        while i < (self.layercount): 
            self.layers[i].weights.mat = np.load(wfile+str(i)+".npy")
            #print(self.layers[i].weights.mat)
            
            self.layers[i].weights.saveToGpu()
            self.layers[i].bias.mat = np.load(bias+str(i)+".npy") 
            self.layers[i].bias.saveToGpu()  
            i=i+1
            
            
    def bulk_injest(self, input_args, target_args):
        self.layers[0].output.bulk_injest(input_args,target_args)
        
    def bulk_print(self):
        self.layers[0].output.bulk_print()
        
    def batch_predict(self, input_args, target_args, input_index_unused):
        ret = self.layers[0].batch_output.batch_injest(input_args,target_args,self.input_index )
        self.batch_target.mat[self.input_index][0] = target_args[0]
        self.input_index = self.input_index + 1
        if self.input_index == self.batch_size:
            self.input_index = 0
        else:
            return False
     
        i = 1
        while i < (self.layercount):               
            self.layers[i].batch_output.compositeActivation(self.layers[i].weights, self.layers[i-1].batch_output, self.layers[i].bias)  
            i = i+1
          
        self.layers[self.last_layer].batch_output.loadFromGpu()      
        # Stats : calculate the error
        i=0
        target_sum = 0
        while i<self.batch_size :
            self.layers[self.last_layer].batch_error_val.mat[0][i] =  self.batch_target.mat[i][0] - self.layers[self.last_layer].batch_output.mat[0][i]
            self.cumulative_error = self.cumulative_error + abs(self.layers[self.last_layer].batch_output.mat[0][i] - self.batch_target.mat[i][0])
            target_sum = target_sum + self.batch_target.mat[i][0]
            i=i+1

        self.layers[self.last_layer].batch_error_val.saveToGpu()
        
        self.data_count = self.data_count +self.batch_size
        self.error_percentage = (self.cumulative_error/self.data_count)*100
        return True
    
    def batch_train(self, input_args, target_args, input_index):
        if self.batch_predict(input_args,target_args,input_index)==False :
            return

        if self.debug==True :
            print(self.data_index,"Batch Input..:", input_args, "Target:", target_args, " : ",self.layers[self.layercount-1].batch_output.getFirstElement())
            print("Weights: ",self.layers[self.last_layer].weights.mat)
            #self.debug = False
            
        i = self.last_layer
        while i>0 : 
            #print("Batch output: ",self.layers[i].batch_output.mat," errorval: ",self.layers[i].batch_error_val.mat)       
            self.layers[i].batch_gradients.compositeDeActivation(self.layers[i].batch_output,self.layers[i].batch_error_val,self.learning_rate, self.layers[i].bias)
            #print("Batch gradints: ",self.layers[i].batch_gradients.mat)
            self.layers[i].weights.compositeMultiplyTranspose(self.layers[i].batch_gradients, self.layers[i-1].batch_output, self.layers[i].weights_delta)

            if (i-1)>0 :
                self.layers[i-1].batch_error_val.multiplyTranspose2(self.layers[i].weights, self.layers[i].batch_error_val)
            i = i-1   
                                            
    def predict(self, input_args, target_args, input_index):
        self.layers[0].output.input_index = input_index
        #self.target_mat.injest(target_args)
                  
        i = 1
        while i < (self.layercount):               
            self.layers[i].output.compositeActivation(self.layers[i].weights, self.layers[i-1].output, self.layers[i].bias)  
            i = i+1
                
        final_output = self.layers[self.last_layer].output.getFirstElement() 
        #self.layers[self.last_layer].error_val.subtract(self.target_mat, self.layers[last_layer].output)
        self.layers[self.last_layer].error_val.subtractScalar(target_args[0], final_output)
        
        # Stats : calculate the error
        self.cumulative_error = self.cumulative_error + abs(final_output - target_args[0])
        self.data_count = self.data_count +1
        self.error_percentage = (self.cumulative_error/self.data_count)*100
        return True
                    
    def train(self, input_args, target_args, input_index):
        if self.predict(input_args,target_args,input_index)==False :
            return

        if self.debug==True :
            print(self.data_index," Input..:", input_args, "Target:", target_args, " : ",self.layers[self.layercount-1].output.getFirstElement())
            print("Weights: ",self.layers[self.last_layer].weights.mat)
            
            
        i = self.last_layer
        while i>0 :  
            #print(" output: ",self.layers[i].output.mat," errorval: ",self.layers[i].error_val.mat)           
            self.layers[i].gradients.compositeDeActivation(self.layers[i].output,self.layers[i].error_val,self.learning_rate, self.layers[i].bias)
            #print("gradints: ",self.layers[i].gradients.mat)
            self.layers[i].weights.compositeMultiplyTranspose(self.layers[i].gradients, self.layers[i-1].output, self.layers[i].weights_delta)

            if (i-1)>0 :
                self.layers[i-1].error_val.multiplyTranspose2(self.layers[i].weights, self.layers[i].error_val)
            i = i-1
            
