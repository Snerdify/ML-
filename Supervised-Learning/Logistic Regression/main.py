import numpy as np

# For this class we need 3 functions : training , prediction and evaluation
class LogisticRegression:
    def __init__(self, learning_rate=0.001 , epoch=1000):
        """
        initialize the learning rate, epoch, weights and bias 
        weights and bias set to None
        """
        self.lr = learning_rate
        self.epoch = epoch
        self.weights = None
        self.bias = None

    def sigmoid(Self ,z):
        """
        Activation Function : Converts real value number into probablity 0 and 1
        sigma(z) = wx + b ; Helps in making binary classfication 
        """
        return 1/1+np.exp(-z)
    
    def loss(self , y_actual , y_pred):
        """
        Loss Function : Binary Cross Entropy Loss -> LogLoss
        Calculates how y_pred stacks over y_actual(Measures how well the model's predictions match the actual labels.)
        This is applicable to only one training example 
        """
        return np.mean(y_actual*np.log(y_pred)+(1-y_actual)*np.log(1-y_pred))
        # to avoid log(0) use 
        # ep = 1e- 9
        # return np.mean(y_actual*np.log(y_pred+ep)+(1-y_actual)*np.log(1-y_pred+ep))



