import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D
import matplotlib.animation as animation
import matplotlib as mpl
import sys

#I will create a function that will be responsible read data from the csv files
## and after that I will normalise this input data and reshape it. I will also initialise the parameters  
def define_params():
    x= pd.read_csv("linearX.csv", header= None )
    y= pd.read_csv("linearY.csv", header= None )
    m,n = x.shape
    #reshape the X and Y vector into m rows and 1 column format
    x= x.values.reshape(m,)
    y= y.values.reshape(m,)
    #normalise the X 
    X = normalize(x,0)
    Y = y
    theta = np.zeros(n+1) #intialise theta as a n+1 matrix of zeroes 
    arr_1 = np.ones(m) #intercept vector x0=1 
    arr_1 = arr_1.reshape(m,1)
    X = X.reshape(m,1)
    X = np.hstack((arr_1,X)) #the x0=1 intercep added in the feature vector 
    return X,Y,theta,m

#create a function that normalises X 

def normalize(input_val, temp):
    mean = np.mean(input_val)
    std  = np.std(input_val)
    input_val = (input_val-mean)/std #z-score formula to carry out the normalisation 
    return input_val


X,Y,theta,m = define_params() #global variables, they are needed throughout the code. 


#to carry out batch gradient descent I have to define a cost function J(theta)
def j_theta(X, Y, theta):
    m = len(X)
    
    # h(0)= Q^T.X 
    # j(0)= (sum of all elemnts m(h0x-y)^2)/2m #
    #loss value= h0x-y
    h = np.dot(X, theta)
    loss = h - Y
    cost = np.sum(loss **2)
    cost = cost/(2*m)
    return (loss,cost)

#defining the function for gradient descent  alpha= learning_rate
def gradient_descent(X,Y,m,theta,alpha):
    xT = X.transpose()
    cost_arr=[]
    theta0=[]
    theta1=[]
    #GD= 0j- aplha/m (summation(h0x-y)x)
    #the second term is the gradient
    #we have to reduce the cost function and descend towards minima so change is J(0) or change in cost= old cost- new cost 
    old_cost= j_theta(X, Y, theta)[1]
    old_cost = old_cost+1
    #iterating over 1lakh points to reach the minimum and we have to reduce the loss
    for i in range(0,100000):
        loss,cost = j_theta(X,Y,theta)
        #stoping criteria: if the change is cost gets below 10^14 break out of the loop and we have a minima and at that minima value of theta0 and theta1
        if(old_cost - cost) < 0.000000000001:  #10^-12
            break;
        cost_arr.append(cost)
        theta0.append(theta[0])
        theta1.append(theta[1])
        
    #gradient and theta updated values
        gradient = np.dot(xT, loss)/m        # gradient= ((change in cost).XT)/m
        theta = theta - alpha*gradient
        old_cost = cost
    
    data = np.row_stack((cost_arr, theta0, theta1)) #combined array of all parameter. 
    return theta,cost,i,cost_arr,data
#Lets plot Jtheta
def j_theta_plot(cost_arr):
    plt.ylabel('J(theta)')
    plt.xlabel('Iterations')
    plt.plot(cost_arr)
    plt.show()

#to plot our hypothesis function or decison boundary 
def hypo(X,Y,final_theta):
    Xaxis = np.linspace(-2,5,10)  #evenly distrubte x-axis
    Yaxis = final_theta[0] + final_theta[1] * Xaxis #contains value of densities
    plt.plot(Xaxis, Yaxis, label='Predicted Density')
    plt.scatter(X[:,1],Y)
    plt.legend(['Hypothesis function'])
    plt.xlabel('Acidty')
    plt.ylabel('Density')
    plt.show()


def Main():
    X,Y,theta,m = define_params()
    alpha = 0.001
    final_theta,final_cost,iterations,cost_list,data = gradient_descent(X,Y,m,theta,alpha)
    print('Final Theta :',final_theta)
    print('Final Cost :',final_cost)
    print('No. of iterations :',iterations)
    print(data[0, 0:1], data[1, 0:1], data[2, 0:1])
    #function to call decision boundary
    hypo(X,Y,final_theta)


Main()