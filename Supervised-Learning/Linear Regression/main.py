import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('housing.csv')
# Lets write the loss func for the linear regression
def loss_func(m,b,points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].RM
        y = points.iloc[i].MEDV 
        total_error += (y-(m*x)-b)**2   # we dont sum it as we are already running a loop
    return total_error/float(len(points))   # we need to return the average error so divide by total number of points
                        
# The loss func is already included in gradient descent algo , so we dont need to calculate it separately     
# it is only written so that we can print loss   

# to implment linear regression from scratch , write a gradient descent function
# GD is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient
# Basically it is a way to find hyperparameters m and b so as to minimize the cost func . 

# STEP 1: Take partial derivative of m and b , to find the largest error . 
# then just reverse it to find the smallest error
# acc to the math , we calculate partial derivative of slope (m) and dist between actual and predicted value of data points

# we will use the formula of gradient descent to calculate the slope and intercept of the line of best fit

# points are all the data points in the dataset
# m_gradient is just DE/Dm
# b_gradient is just DE/Db

def gradient_descent(m_current , b_current , points , L):
    m_gradient = 0 
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].RM
        y = points.iloc[i].MEDV
        m_gradient = -(2/n) * x *(y-(m_current*x + b_current))
        b_gradient = -(2/n) * (y-(m_current*x + b_current))

        new_m = m_current - L * m_gradient
        new_b = b_current - L * b_gradient

        return new_m , new_b
    
m=0
b=0
L=0.0001
epochs = 300

for i in range(epochs):
    if 1%50==0:
        print(f"Epochs : {i}")
    m_new , b_new  = gradient_descent(m,b,data,L)

print(m_new , b_new)  # 1.3553374233128836 0.2061349693251534

plt.scatter(data.RM , data.MEDV, color ="black")  # scatter plot of data points
# plot the line of best fit












