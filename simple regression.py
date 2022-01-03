import numpy as np
import matplotlib.pyplot as plt 

def estimate_coef(x,y):
    n=np.size(x)  #number of points
    
    
    #mean of x and y vector
    m_x=np.mean(x)
    m_y=np.mean(y)
    
    #calculate cross deviation
    ss_xy=np.sum(y*x)-n*m_x*m_y
    ss_xx=np.sum(x*x)-n*m_x*m_x
    
    #calculating regression coeff
    b_1=ss_xy/ss_xx
    b_0=m_y-b_1*m_x
    
    return (b_0,b_1)

def plot_regression_line(x,y,b):
    #plotting the points as scatter
    plt.scatter(x,y,color="m",marker="o",s=30)
    
    
    
    #predicted respose vector
    y_pred=b[0]+b[1]*x
    
    
    #plotting the regression line:
    plt.plot(x,y_pred,color="g")
    
    
    #put labels
    plt.xlabel('x')
    plt.ylabel('y')
    
    
    
    #show plot
    plt.show()
    


def main():
    #data
    x=np.array([0,1,2,3,4,5,6,7,8,9])
    y=np.array([1,3,3,4,5,8,7,8,9,12])
    
    #stimate coeff
    b=estimate_coef(x,y)
    print("Estimated coefficients:\nb_0={}\nb_1={}".format(b[0],b[1]))
    
    #plot regression line
    plot_regression_line(x,y,b)
    
if __name__=="__main__":
    main()
    
    
