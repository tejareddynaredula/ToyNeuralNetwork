
import pandas as pd
import random
import numpy as np 
import math
import sys
#from numba import vectorize
#from numba import cuda, float32
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit

kernel_code_template = """
#include <stdio.h>
#include<stdlib.h>
__global__ void MatrixAddKernel(int rows, int columns, float *a, float *b, float *c)
{
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;

    if (tx < rows  && ty < columns )
    c[tx * columns  + ty] = a[tx * columns  + ty]  + b[tx * columns  + ty] ;

}
__global__ void MatrixSubtractKernel(int rows, int columns, float *a, float *b, float *c)
{
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;

    if (tx < rows  && ty < columns )
    c[tx * columns  + ty] = a[tx * columns  + ty]  - b[tx * columns  + ty] ;

}

__global__ void CompositeDeActivationKernel(int rows, int columns, float *a, float *b, float scalar, float *c ,float *bias)
{
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    
    // copy + deactivation+MultiplyScalar   self=A, deactivation , self= self*B*s
    float Output = 0;
    
    if (tx < rows  && ty < columns ) {
    // 1.copy
        Output = a[tx * columns  + ty];
        
    //2. Deactivation
        Output = Output * (1-Output) ;
        
    //3. Multiply with b and scalar
        Output = (Output) * (b[tx * columns  + ty]) * (scalar);
        c[tx * columns  + ty] = Output;
    
    //4. Add to second output(bias)
        bias[tx * columns  + ty] = bias[tx * columns  + ty] + Output;
    }
}
__global__ void CompositeBatchActivationKernel(int rows, int common_cols, int columns, float *a, float *b, float *c, float*d, int batch_size)
{
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    
    //  # multiply + add + mapActivation : output = (A*B) + C  , activation
    // A = weights B =old output   C= bias
    // each thread picks i and j of the matrix
    //        self.mat = np.matmul(A.mat,B.mat)
    //        self.add(self,C)
    //        self.mapActivation()

    float Output = 0;

    if (tx < rows  && ty < columns ) {   
       // 1.Multiplication
       for (int k = 0; k < common_cols; ++k) {
          float Aelement = a[tx * common_cols + k];
          float Belement = b[k * columns + ty];
          Output += Aelement * Belement;
       } 
 
       // 2. Addition
        Output = Output  + c[tx * columns ] ;
       
       //3. Aactivation 
        d[tx * columns + ty] = 1/(1+ exp(-Output)) ; 
    }
}
__global__ void CompositeActivationKernel(int rows, int common_cols, int columns, float *a, float *b, float *c, float*d)
{
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    
    //  # multiply + add + mapActivation : output = (A*B) + C  , activation
    // each thread picks i and j of the matrix
    //        self.mat = np.matmul(A.mat,B.mat)
    //        self.add(self,C)
    //        self.mapActivation()

    float Output = 0;

    if (tx < rows  && ty < columns ) {    
    // 1.Multiplication
       for (int k = 0; k < common_cols; ++k) {
          float Aelement = a[tx * common_cols + k];
          float Belement = b[k * columns + ty];
          Output += Aelement * Belement;
       } 
 
    // 2. Addition
       Output = Output  + c[tx * columns  + ty] ;
       
    //3. Aactivation 
      d[tx * columns + ty] = 1/(1+ exp(-Output)) ; 
    }
}
__global__ void CompositeTransposeMulKernel(int rows, int common_cols, int columns, float *a, float *b, float *c, float *d)
{
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    
    //MultiplyTranspose + add ,  output= A*B.T + C
    float Output = 0;

    if (tx < rows  && ty < columns ){
       for (int k = 0; k < common_cols; ++k) {
          float Aelement = a[tx * common_cols + k];
          float Belement = b[ty * common_cols + k];
          Output += Aelement * Belement;
       }
       //d[tx * columns + ty] = Output + c[tx * columns + ty];
       // NOT Needed c[tx * columns + ty] = Output
       d[tx * columns + ty] =  d[tx * columns + ty] + Output;
    }
}
__global__ void Transpose2MatrixMulKernel(int rows, int common_cols,  int columns, float *a, float *b, float *c)
{
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;

    float Pvalue = 0;

    if (tx < rows  && ty < columns ) {
       for (int k = 0; k < common_cols; ++k) {
          float Aelement = a[k * rows + tx];
          float Belement = b[k * columns + ty];
          Pvalue += Aelement * Belement;
       }
       c[tx * columns + ty] = Pvalue;
    }
}
__global__ void Transpose2MulScalarKernel(int rows, int common_cols,  int columns, float *a, float b, float *c)
{
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;

    float Pvalue = 0;

    if (tx < rows  && ty < columns ) {
       for (int k = 0; k < common_cols; ++k) {
          float Aelement = a[k * rows + tx];
          Pvalue += Aelement * b;
       }
       c[tx * columns + ty] = Pvalue;
    }
}

__global__ void TransposeMatrixMulKernel(int rows, int columns, float *a, float *b, float *c)
{
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;

    float Pvalue = 0;

    if (tx < rows  && ty < columns ){
       for (int k = 0; k < columns; ++k) {
          float Aelement = a[ty * columns + k];
          float Belement = b[tx * columns + k];
          Pvalue += Aelement * Belement;
       }
       c[ty * columns + tx] = Pvalue;
    }
}

"""

class Matrix:
    #activation = "sigmoid"
    gpu_matrixadd = 1
    matrix_initialised = False      
    class InnerMatrix:
        def __init__(self,rows,cols):
            self.rows = np.int32(rows)
            self.cols = np.int32(cols)
            
    def __init__(self, rows,cols,gpu_enabled,vector, activation="sigmoid"):
        if True and Matrix.matrix_initialised == False :
            Matrix.matrix_initialised  = True
            kernel_code = kernel_code_template % {
                }
            mod = compiler.SourceModule(kernel_code)
            
            Matrix.gpu_matrixaddkernel = mod.get_function("MatrixAddKernel")
            Matrix.gpu_matrixSubtractkernel = mod.get_function("MatrixSubtractKernel")
            Matrix.gpu_multiplyTrans2matrixkernel = mod.get_function("Transpose2MatrixMulKernel")
            Matrix.gpu_multiplyTrans2Scalarkernel = mod.get_function("Transpose2MulScalarKernel")
                        
            Matrix.gpu_compositeActivationkernel = mod.get_function("CompositeActivationKernel")
            Matrix.gpu_compositeBatchActivationkernel = mod.get_function("CompositeBatchActivationKernel")
            Matrix.gpu_compositeDeactivationkernel = mod.get_function("CompositeDeActivationKernel")
            Matrix.gpu_compositeTransposeMulkernel = mod.get_function("CompositeTransposeMulKernel")
            
            print("Initialised KERNELS for GPU  version 1.00")

        self.gpu_enabled = gpu_enabled
        self.im_list = [ ]
        self.inner_enable = False
        self.input_index = 0
        self.activation = activation
        if vector is not None:
            self.rows = np.int32(1)
            self.cols = np.int32(len(vector))
            self.mat = np.array([[random.random() for col in range(self.cols)] for row in range(self.rows)],dtype='f')
            if True:
                k=1
                for i in range(self.rows):
                    for j in range(self.cols):
                        self.mat[i][j] = k
                        self.mat[i][j] = 1
                        k=k+1
                        
            for i in range(self.cols):
                self.mat[0][i] = vector[i]
        else:
            self.rows = np.int32(rows)
            self.cols = np.int32(cols)
            self.mat = np.array([[random.random() for col in range(self.cols)] for row in range(self.rows)],dtype='f')
            #TODO :change below type to float32 for gpu
            #initialise the weights accordingly
            self.mat = np.random.randn(rows,cols).astype('f')*np.sqrt(2/cols)
            
            # Used for unit testing
            if False:
                k=1
                for i in range(self.rows):
                    for j in range(self.cols):
                        self.mat[i][j] = k*0.25
                        k=k+1
                
        if self.gpu_enabled : 
            self.saveToGpu()
            self.gpu_blockx = 25
            self.gpu_blocky = 25
            self.gpu_block= (np.int(self.gpu_blockx ),np.int(self.gpu_blocky),np.int(1))
            self.gpu_grid = (np.int(self.rows/self.gpu_blockx)+1,np.int(self.cols/self.gpu_blockx)+1,np.int(1))
        else:
            self.mat_gpu =0
    
    def saveToGpu(self):
        if self.gpu_enabled :
            self.mat_gpu = gpuarray.to_gpu(self.mat)

    def loadFromGpu(self):
        if self.gpu_enabled :
            self.mat_gpu.get(self.mat)
                        
    def getFirstElement(self):
        if self.gpu_enabled :
            self.mat_gpu.get(self.mat)
            return self.mat[0][0]
        else:
            return self.mat[0][0] 

    def batch_injest(self, input_vector, target, index): 

        for i in range(len(input_vector)):
            self.mat[i][index] = input_vector[i]
            
        if self.gpu_enabled and index==(self.cols-1):
            self.mat_gpu.set(self.mat)
            return True
        else:
            if index==(self.cols-1):
                return True
            else:
                return False
            
               
    def bulk_injest(self, vector, target):
        if self.cols > 1:
            print("ERROR in Injest: More then columns") 
        
        inner_matrix = Matrix.InnerMatrix(len(vector),1)
        inner_matrix.mat = np.array([[random.random() for col in range(inner_matrix.cols)] for row in range(inner_matrix.rows)],dtype='f')
        for i in range(inner_matrix.rows):
            inner_matrix.mat[i][0] = vector[i]  
        inner_matrix.target = target
        self.inner_enable = True
        
        if self.gpu_enabled :
            inner_matrix.mat_gpu = gpuarray.to_gpu(inner_matrix.mat) 
            inner_matrix.mat_gpu.set(inner_matrix.mat)  
        else:
            inner_matrix.mat_gpu = 0
            
        self.im_list.append(inner_matrix) 
        
    def bulk_print(self):
        for i in range(len(self.im_list)):
            print(i," ::::" ,self.im_list[i].mat)
            
                      
    def injest(self, X ):
        if self.cols > 1:
            print("ERROR in Injest: More then columns") 
        for i in range(self.rows):
            self.mat[i][0] = X[i]
            
        if self.gpu_enabled :
            self.mat_gpu.set(self.mat)
  
    def average(self, X):
        if self.gpu_enabled :
            #TODO: need to implement
            self.mat[0][0]=0   
        else: 
            for i in range(X.rows):
                self.mat[i][0]=0
                for j in range(X.cols):           
                    self.mat[i][0] = self.mat[i][0] + X.mat[i][j] 
                self.mat[i][0] = self.mat[i][0]/X.cols             
    
    def add(self, X,Y):
        if self.gpu_enabled :
            self.gpu_matrixaddkernel(
                self.rows,self.cols,
                X.mat_gpu, Y.mat_gpu, 
                self.mat_gpu, 
                block = self.gpu_block, grid = self.gpu_grid,
                )
        else:            
            self.mat = X.mat + Y.mat
        
    def mapActivation(self):
        if self.gpu_enabled :
            print("ERROR : Not expected map activation in gpu ")
            sys.exit()
            
        if self.activation == "sigmoid" :
            self.mat = 1 / (1 + np.exp(-self.mat))
        elif self.activation == "leaky_relu":
            #self.mat = np.where(self.mat > 0, self.mat, self.mat * 0.01)
            self.mat = np.where(self.mat > 0, self.mat, 0)
            return
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.mat[i][j] <0 :
                        #self.mat[i][j] = self.mat[i][j] * 0.01
                        self.mat[i][j] = 0

            
        else:
            self.mat = np.tanh(self.mat)
            
    def mapDeActivation(self):
        if self.gpu_enabled :
            print("ERROR : Not expected map Deactivation in gpu ")
            sys.exit()
            
        if self.activation == "sigmoid" :
            self.mat =  self.mat * (1 - self.mat)
        elif self.activation == "leaky_relu":
            # TODO: need to use numpy function instead of iteration
            self.mat = np.where(self.mat > 0, 1, 0)
            return
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.mat[i][j] >0 :
                        self.mat[i][j] = 1
                    else:
                        self.mat[i][j] = 0
        else:
            self.mat = (1.0 - (self.mat*self.mat))  
            
    def compositeActivation(self,Weigts,Output_arg,Bias):
        # multiply + add + mapActivation : output = (A*B) + C
        if (Output_arg.inner_enable):
            index = Output_arg.input_index
            Output_mat = Output_arg.im_list[index].mat
            Output_mat_gpu = Output_arg.im_list[index].mat_gpu
        else:
            Output_mat = Output_arg.mat
            Output_mat_gpu = Output_arg.mat_gpu
            
        if self.gpu_enabled :
            self.gpu_compositeActivationkernel(
                self.rows, Weigts.cols, self.cols,
                Weigts.mat_gpu, Output_mat_gpu,Bias.mat_gpu, 
                self.mat_gpu, 
                block = self.gpu_block, grid = self.gpu_grid,
                )
        else:
            self.mat = np.matmul(Weigts.mat,Output_mat)
            self.add(self,Bias)
            self.mapActivation()
                    
    def compositeDeActivation(self, A,B,s, bias):
        # copy + deactivation+MultiplyScalar   self=A, deactivation , self= self*B*s
        if self.gpu_enabled :
            v = np.float32(s)
            self.gpu_compositeDeactivationkernel(
                self.rows,self.cols,
                A.mat_gpu, B.mat_gpu, v , 
                self.mat_gpu, 
                bias.mat_gpu,
                block = self.gpu_block, grid = self.gpu_grid,
                )
        else:
            self.mat = A.mat
            self.mapDeActivation()
            self.mat = self.mat*B.mat*s
            for i in range(self.rows):
                temp_sum=bias.mat[i][0]
                for j in range(self.cols):
                    temp_sum=temp_sum+self.mat[i][j]
                bias.mat[i][0]=temp_sum/self.cols
            #bias.add(bias,self)
        
    def compositeMultiplyTranspose(self, A,B_arg,C):
        if A.cols != B_arg.cols:
            print("ERROR in MultiplicationTranspose")
        # MultiplyTranspose + add ,  output= A*B.T + C
        if (B_arg.inner_enable):
            index = B_arg.input_index
            B_mat = B_arg.im_list[index].mat
            B_mat_gpu = B_arg.im_list[index].mat_gpu
        else:
            B_mat = B_arg.mat
            B_mat_gpu = B_arg.mat_gpu
            
        if self.gpu_enabled :
            self.gpu_compositeTransposeMulkernel(
                self.rows, A.cols, self.cols,
                A.mat_gpu, B_mat_gpu, C.mat_gpu, 
                self.mat_gpu, 
                block = self.gpu_block, grid = self.gpu_grid,
                )
        else:        
            #C.multiplyTranspose(A,B)
            C.mat = np.matmul(A.mat,B_mat.T)
            self.add(self, C)
     
                    
    def subtract(self, X,Y):
        if self.gpu_enabled :
            self.gpu_matrixSubtractkernel(
                self.rows,self.cols,
                X.mat_gpu, Y.mat_gpu, 
                self.mat_gpu, 
                block = self.gpu_block, grid = self.gpu_grid,
                )     
        else:           
            self.mat = X.mat - Y.mat
            
    def subtractScalar(self, s,y):
        if self.gpu_enabled :
            self.mat[0][0] = s - y
            self.saveToGpu()   
        else:           
            self.mat[0][0] = s - y

    def multiplyTranspose2(self, X,Y):
        if X.rows != Y.rows:
            print("ERROR in MultiplicationTranspose2")
            
        if self.gpu_enabled :
            #print(" common rows: ",X.rows)
            self.gpu_multiplyTrans2matrixkernel(
                self.rows,X.rows,self.cols,
                X.mat_gpu, Y.mat_gpu,
                self.mat_gpu,
                block = self.gpu_block, grid = self.gpu_grid,
                )
        else:
            self.mat = np.matmul(X.mat.T,Y.mat) 
            
    def multiplyTranspose2Scalar(self, X, s):
            
        if self.gpu_enabled :
            #print(" common rows: ",X.rows)
            v = np.float32(s)
            self.gpu_multiplyTrans2Scalarkernel(
                self.rows,X.rows,self.cols,
                X.mat_gpu, v,
                self.mat_gpu,
                block = self.gpu_block, grid = self.gpu_grid,
                )
        else:
            self.mat = X.mat.T*s
                       
    def printMat(self,s):
        if self.gpu_enabled :
            print("gpu Matrix: ",s,self.mat_gpu.get())
        else:
            print("Matrix: ", s,self.mat)
                  
    def NOTINUSEcompositeBatchActivation(self,Weigts,Output,Bias):
        # multiply + add + mapActivation : output = (A*B) + C
        # newoutput = weights * old_output + Bias
        batch_size =  np.float32(Output.cols)
        if self.gpu_enabled :
            self.gpu_compositeBatchActivationkernel(
                self.rows, Weigts.cols, self.cols,
                Weigts.mat_gpu, Output.mat_gpu,Bias.mat_gpu, 
                self.mat_gpu, batch_size,
                block = self.gpu_block, grid = self.gpu_grid,
                )
        else:
            self.mat = np.matmul(Weigts.mat,Output.mat)
            for i in range(self.rows):
                for j in range(self.cols): 
                    self.mat[i][j] = self.mat[i][j] + Bias.mat[i][0]
                    self.mat[i][j] = 1 / (1 + np.exp(-self.mat[i][j]))
                    
            #self.mapActivation()