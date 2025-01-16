# loss func helps us understand how good our preds are
# Also tells us how to adjust the parameters of out network

import numpy as np

from lib.tensor import Tensor 
# the loss func is squared loss func
class Loss:
    def loss(self , predicted:Tensor , actual:Tensor)-> float:
        return np.sum((predicted-actual)**2)   
    
    def grad(self , predicted:Tensor , actual:Tensor)-> Tensor:
        return 2 *(predicted-actual)
