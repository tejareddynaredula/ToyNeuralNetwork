 
# Performance of ToyNeuralNet on various Platforms:
A "toy neural network" generally refers to a simple implementation of a neural network, this is used for educational or demonstration purposes. Neural networks are a type of machine learning algorithm modeled after the structure and function of the human brain, and they are used for a wide range of tasks such as image classification, speech recognition, and natural language processing.

A toy neural network typically has only a few layers and a limited number of neurons, and it may not have the complexity or performance of larger, production-ready neural networks. However, building a toy neural network can be a great way to learn about the basics of neural network architecture, activation functions, loss functions, and backpropagation, which are fundamental concepts in deep learning.

ToyNeuralNet is Neuralnet implemented from scratch without using any ML library. The goal of the project is to measure the performance of NeuralNet on various platforms like python vs golang, cpu vs gpu, network size, different python implementation , batch size, trainning vs inference..etc

Following are different performance experiments of ToyNeuralnet used for Inference and Training purposes:

- ToyNeuralNet using different versions of Python:
    - using python List : very slow (260sec)
    - using python Arrays: Better then List (160sec)
    - using numpy library: Best of all the three (95sec) 
    - Summary: Numpy is best of all the three(python list,python array and numpy). Reason: Numbpy Array are implemented natively, due to this matrix multiplication is faster.
    - using GPU : better then numpy when the layer contain large number of neurons.
- ToyNeuralNet using CPU versus GPU with python:
    - Small number of neurons vs large number of neurons per layer:
      -  CPU as better latency when compare to GPU for small number of neurons. for large neurons GPU is better. The reason is overhead in submitting the job to GPU, overhead of CPU and GPU communication is large for small neuralnets when compare to large neural net.
    - Workload: Training vs predict:  
      -  GPU provides better latency in Training  when compare to Predict. Reason: Predict contain mXn by mX1 multiplication, but Trainning need mXn by mXn mulitplication , and also update of weights.
    -  Batch vs Without Batch: 
    	 - Batch is good for GPU and CPU. As batch level increases, the efficiency of cpu parallelism goes up.  
- ToyNeuralNet using golang vs Python: Partially done.
- ToyNeuralNet inside the linux kernel especially for inference: In streaming inference  there will be lot of network IO, means there there will lot of system calls and memorycopy. Pushing ToyNeuralNet inside the linux kernel avoid system call overhead and gives better memory copy.  
    

# Test Results 

- Data Set : Iris flower Dataset.
- Large-NeuralNet:  with large number of neurons: Network size: [4,1600,2600,1600,2500,1]
- Small-NeuralNet:  with small number of neurons: Network size: [4,3,4,5,1]
- Test script: test_NN.py

 
<table border=1>
<thead>
<tr>
<th>Test-no</th>
<th>Description</th>
<th>Result</th>
</tr>
<tr>
<th>1</th>
<th> Large-NeuralNet Trainning with CPU batch=3
   </th>
<th>Time taken by Trainning=24sec  cpu utilization=600% gpu utilization=0%.  Summary: CPU is slower when compare to GPU.</th>
</tr>
<tr>
<th>2</th>
<th> Large-NeuralNet Trainning with GPU batch=3
</th>
<th>Time taken by Trainning=9sec 
 cpu utilization=100% gpu=100%. Summary: Test on GPU is faster when compare to CPU. 
</th>
</tr>

<tr>
<th>3</th>
<th>Large-NeuralNet Trainning with CPU batch=10
   </th>
<th>Time taken by Trainning=9sec  cpu utilization=600% gpu utilization=0%</th>
</tr>
<tr>
<th>4</th>
<th>Large-NeuralNet Trainning with GPU batch=10
</th>
<th>Time taken by Trainning=5.4sec 
 cpu utilization=100% gpu=100%
</th>
</tr>
<tr>
<th>5</th>
<th>Large-NeuralNet Prediction/Inference with GPU batch=10
</th>
<th>  CPU is slightly faster then GPU, Reason why GPU is not faster: a) Matrix multiplication used in Inference is MxN by MX1  instead of MXN by MXN used in trainning ,  for large batching GPU becomes better. 2) In Inference there will not be much memory writes so gpu computations are less.
</th>
</tr>
<tr>
<th>6</th>
<th>Small-NeuralNet Trainning with GPU Vs CPU
</th>
<th>CPU latency is much better when compare to GPU, GPU as overhead of loading the data from main memory to GPU and viceversa for a very small computation. 
</th>
</tr>
</tbody></table>

# overhead of Syscalls 

 ```
 CPU TEST: 600% CPU : most of time spend in matrix multiplication
 system calls per second in CPU test: 620k /sec
 % time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
 55.93    1.908009           2   1240313           sched_yield
 39.68    1.353411       22939        59           poll
  4.14    0.141347        1178       120           clock_gettime
  0.25    0.008448        8448         1           restart_syscall
  0.00    0.000000           0        50         6 futex
------ ----------- ----------- --------- --------- ----------------

GPU TEST: 100% CPU  :  Most of the time spend in IO with GPU due to this time for futex  are high.
 system calls per second in CPU test: 100k /sec, Here mostly IO instensive with GPU.
% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
 41.52    5.524696       21923       252           poll
 40.11    5.337111       53910        99        49 futex
 17.40    2.314710           1   2109500           clock_gettime
  0.68    0.090668       45334         2         1 restart_syscall
  0.30    0.039534           1     36481           getpid
  0.00    0.000029           3        10           write
------ ----------- ----------- --------- --------- ----------------
100.00   13.306748               2146344        50 total
```

GPU spec:

```
For GPU test
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 390.48                 Driver Version: 390.48                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GT 1030     Off  | 00000000:01:00.0  On |                  N/A |
| 55%   54C    P0    N/A /  30W |    495MiB /  2000MiB |     99%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1242      G   /usr/lib/xorg/Xorg                            18MiB |
|    0      1284      G   /usr/bin/gnome-shell                          48MiB |
|    0      3392      G   /usr/lib/xorg/Xorg                           125MiB |
|    0      3548      G   /usr/bin/gnome-shell                          80MiB |
|    0      6291      G   ...-token=E69B0CC02C6F5687670BDDD2298FD240    80MiB |
|    0      7004      C   python                                       131MiB |
+-----------------------------------------------------------------------------+
```
top:  for CPU TEST : cpu consumption: 600%

```
top - 11:28:28 up  2:28,  3 users,  load average: 1.49, 1.98, 2.02
Tasks: 302 total,   2 running, 240 sleeping,   0 stopped,   0 zombie
%Cpu(s): 99.1 us,  0.9 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.1 si,  0.0 st
KiB Mem : 16354764 total, 12074680 free,  1796652 used,  2483432 buff/cache
KiB Swap:        0 total,        0 free,        0 used. 14183740 avail Mem 

  PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND                                                                                                                                             
 6930 a         20   0 13.431g 339968 121916 R 595.7  2.1   1:18.84 python
 ```
