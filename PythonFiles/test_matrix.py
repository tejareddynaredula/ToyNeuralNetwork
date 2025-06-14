from Matrix import *
import time
import cProfile

pr = cProfile.Profile()
pr.enable()

# Here maximum 32X32=1024 of blocksize is allowed 
'''
root@HomeServer:/data/gpu_test# python ./test_matrix.py 
GPU Time taken:  48.32857346534729  /* cpu=100% */
CPU Time taken:  76.59190392494202   /* cpu=600% */
'''

r=5
c=5

mat0 = Matrix(3,1,True, None)

mat1 = Matrix(3,4,True, None)
mat2 = Matrix(4,1,True, None)

mat3 = Matrix(3,1,True, None)
mat4 = Matrix(3,1,False, None)

mat00 = Matrix(r,c,False, None)
mat11 = Matrix(r,c,False, None)
mat22 = Matrix(r,c,False, None)

mat5 = Matrix(r,c,False, None)

input = [6.4,2.8,5.6,2.2]
mat2.injest(input)

loops = 1000000
loops = 1
print("mat1:",mat1.mat)
print("mat2:",mat2.mat)
print("mat3: ",mat3.mat_gpu.get())
start = time.time()
i =0
while i<loops:  
    #mat3.add(mat1,mat2)
    #mat3.compositeDeActivation(mat0, mat1, 2.5)
    
    mat3.compositeActivation(mat1,mat2,mat0)

    #mat3.compositeMultiplyTranspose(mat0, mat1, mat2)
    
    #mat3.multiplyTranspose2( mat1, mat2)
    #mat3.injest(input)
    i=i+1
end = time.time()
print("GPU Time taken: ",(end-start))

start = time.time()
i =0
while i<loops:  

    
    #mat4.add(mat1,mat2)
    #mat4.compositeDeActivation(mat0, mat1, 2.5)
    mat4.compositeActivation(mat1,mat2,mat0)
    #mat4.compositeMultiplyTranspose(mat0, mat1, mat2)

    
    #mat4.multiplyTranspose2( mat1, mat2)
    #mat4.injest(input)

    i=i+1
end = time.time()
#print(" mat4 single element: ",mat4.getFirstElement())
#print(" mat3 single element: ",mat3.getFirstElement())
print("CPU Time taken:. transpose1.1. ",(end-start))

print ("-" * 80)
print ("Matrix GPU --- :")
print (mat3.mat_gpu.get())
t=mat3.mat_gpu.get()
print(" type: ",type(t))
print ("Matrix CPU --- :")
print (mat4.mat)
pr.disable()
#pr.print_stats()