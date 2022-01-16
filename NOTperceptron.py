import numpy as np


def unitStep(v):
    if v>=0:
        return 1
    else:
        return 0
    
#perceptron model

def perceptron(x,w,b):
    v=np.dot(w,x)+b
    y=unitStep(v)
    return y

#not logic function
#w=-1,b=0.5
def Not_logicFunction(x):
    w=-1
    b=0.5
    return perceptron(x,w,b)

#testing the perceptron
test1=np.array(1)
test2=np.array(0)

print("NOT({})={}".format(1,Not_logicFunction(test1)))
print("NOT({})={}".format(0,Not_logicFunction(test2)))
