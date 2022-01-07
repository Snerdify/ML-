#to do all the math 

import numpy as np
#to plot our data and visualize it
from matplotlib import pyplot as plt
%matplotlib inline

#step1 -define our data

#input data of the form(x,y,bias)

x=np.array([[-2,-4,3],[1,2,3],[1,6,-1],[2,4,-1],[6,2,-1]])

#each of the 5 inputs has a output label. First 2 are labeled -1 and next 3 are labelled 1

y=np.array([-1,-1,1,1,1])

#now plot these examples on a 2D graph
for d,sample in enumerate(x):
    #plot the negative first 2 samples
    if (d<2):
        plt.scatter(sample[0],sample[1],s=120,marker='_',linewidth=2)
        
    #plot the positive next 3 samples
    else:
        plt.scatter(sample[0],sample[1],s=120,marker='+',linewidth=2)
        
#print the hyperplane sepearting two classes

plt.plot([-2,6],[6,0.5])
        
