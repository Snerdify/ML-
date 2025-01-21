import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('housing.csv')

# to implment linear regression from scratch , write a gradient descent function

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
        m_new , b_new  = gradient_descent(m,b,data,L)

    print(m_new , b_new)

plt.scatter(data.RM , data.MEDV, color ="black")
plt.show()










