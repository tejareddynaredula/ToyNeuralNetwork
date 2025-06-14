from NeuralNetV4 import *
import cProfile
import sys
import time

pr = cProfile.Profile()
pr.enable()

batch = True
batch_len=10
nn = NeuralNet([4,1600,2600,1600,2500,1],gpu_enabled=False,name="testsmall",load=False,batch_size=batch_len,learning_rate=0.005)
#nn = NeuralNet([4,5600,5600,5600,5500,1],gpu_enabled=True,name="testLarge",load=False,batch_size=batch_len)
#nn = NeuralNet([4,3,4,5,1],gpu_enabled=False,name="testsmall",load=False,batch_size=batch_len)
#nn = NeuralNet([4,3,4,5,1],False)
data = pd.read_csv("./iris_training.csv")
#print (data)
sepallength = data.SepalLength.values.tolist()
sepalwidth = data.SepalWidth.values.tolist()
petallength = data.PetalLength.values.tolist()
petalwidth = data.PetalWidth.values.tolist()
types = data.types.values.tolist()

datalen = len(sepallength)
i=0
while i< datalen :
    if types[i] == 2:
        types[i] = 0.5
    nn.bulk_injest([sepallength[i],sepalwidth[i],petallength[i],petalwidth[i]],types[i])
    i=i+1
    
#nn.bulk_print()
#sys.exit()
#print("slots: ",nn.__dict__)

iteration=0
nn.debug=False
start = time.time()
while iteration< 260 :
    i=0
    if iteration%10 == 0:
        print(iteration," load: ",nn.load, "samples: ",nn.data_count,"batch:",batch," Batch_size: ",nn.batch_size," Error Percentage: ",nn.error_percentage)
        end = time.time()
        print("Time taken: ",(end-start))
        start = time.time()
        nn.data_count =0
        nn.cumulative_error =0
        #nn.debug=True
        
    while i< datalen :
        if types[i] == 2:
            types[i] = 0.5
        if nn.load:
            if batch:
                nn.batch_predict([sepallength[i],sepalwidth[i],petallength[i],petalwidth[i]],[types[i]],i)
            else:
                nn.predict([sepallength[i],sepalwidth[i],petallength[i],petalwidth[i]],[types[i]],i)
        else: 
            if batch:
                nn.batch_train([sepallength[i],sepalwidth[i],petallength[i],petalwidth[i]],[types[i]],i)
            else: 
                nn.train([sepallength[i],sepalwidth[i],petallength[i],petalwidth[i]],[types[i]],i)
        i=i+1
        if i>2 and nn.debug==True:
            sys.exit()
    if iteration%2000 == 0:
        nn.debug=False
    iteration=iteration+1

if nn.load==False:
    nn.saveNeuralNet()    
pr.disable()
#pr.print_stats()
