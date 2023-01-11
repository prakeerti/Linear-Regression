import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D
import matplotlib.animation as animation
import matplotlib as mpl


#read_data() is a function which reads the date from input csv files.
#It normalizes the data and change the shapes of data
def read_data():
    acidity = pd.read_csv("linearX.csv", header=None)
    density = pd.read_csv("linearY.csv", header=None)
    m,n = acidity.shape
    acidity = acidity.values.reshape(m,)
    density = density.values.reshape(m,)
    #feature is normalized here
    X = normalization(acidity,0)
    Y = density
    theta = np.zeros(n+1)
    ones_arr = np.ones(m)
    ones_arr = ones_arr.reshape(m,1)
    X = X.reshape(m,1)
    #x0=1 are added here in the features vector
    X = np.hstack((ones_arr,X))
    return X,Y,theta,m


#normalization is a function which normalizes the data, and we use it to normalize onlu the features and no the lables

def normalization (input_arr, temp):
    mean = np.mean(input_arr)
    std  = np.std(input_arr)
    med  = np.median(input_arr)
    print("mean of data :{}, standard deviation :{}, median of data :{}".format(mean,std,med))
    input_arr = (input_arr-mean)/std
    return input_arr


#X,Y,theta,m are the global variables which are used everywhere in the program
X,Y,theta,m = read_data()


#cost function calculates the cost based on given the X,Y and theta parameters
def cost_function(X, Y, theta):
    # print('shapes ',X.shape,Y.shape,theta.shape)
    m = len(X)
    #hypothesis is calculated here
    hypothesis = np.dot(X, theta)
    loss = hypothesis - Y
    cost = np.sum(loss **2)
    cost = cost/(2*m)
    return (loss,cost)



#gradient_descent is the main
def gradient_descent(X,Y,m,theta,learning_rate):
    xTrans = X.transpose()
    cost_list=[]
    theta0_list=[]
    theta1_list=[]
    #cost function returns loss and cost and we need only cost
    pre_cost=cost_function(X, Y, theta)[1]
    pre_cost = pre_cost+1
    for i in range(0,100000):
        loss,cost = cost_function(X,Y,theta)
        #stoping criteria
        if(pre_cost - cost) < 0.00000000000001:  #10^-14
            break;
        cost_list.append(cost)
        theta0_list.append(theta[0])
        theta1_list.append(theta[1])
        #calculates graidient here
        gradient = np.dot(xTrans, loss)/m       
        #the new value of theta is updated here    
        theta = theta - learning_rate*gradient
        pre_cost = cost
    #data is the array of of 3rows which will be used to plot contour and 3D mesh
    data = np.row_stack((theta0_list, theta1_list,cost_list))
    return theta,cost,i,cost_list,data



#this function draws the linear decision boundary
def plot_decision_boundary(X,Y,final_theta):
    print("points of line are added here")
    lineX = np.linspace(-2,5,10)
    lineY = final_theta[0] + final_theta[1] * lineX
    plt.plot(lineX, lineY, '-r', label='Predicted Density')
    plt.scatter(X[:,1],Y)
    plt.legend(['Hypothesis function','Data'])
    print('labels are set here')
    plt.xlabel('Acidty of wine')
    plt.ylabel('Density of wine')
    plt.show()


#
def Gen_Data_3dMesh(X,Y,m,final_theta):
    # n1_1, n1_2 = final_theta[0]-1.0, final_theta[1]+1.0
    theta0=np.linspace(0.0, 2.0, m)
    n2_1, n2_2 = final_theta[1]-1.0, final_theta[1]+1.0
    theta1=np.linspace(n2_1, n2_2, m)
    x_mesh, y_mesh = np.meshgrid(theta0, theta1)
    z_mesh=np.zeros(10000)
    z_mesh = z_mesh.reshape(m,m)
    #3D mesh data generated
    print('3D Mesh is created here')
    for i in range(m):
        for j in range(m):
            loss,cost = cost_function(X,Y,np.array([x_mesh[i][j],y_mesh[i][j]]))
            z_mesh[i][j] = cost
    return x_mesh,y_mesh,z_mesh 


#animate_line function to animate the line in 3D Mesh
def animate_line(i,lines,data):
    lines.set_data(data[0:2, :i])
    lines.set_3d_properties(data[2, :i])
    lines.set_marker("o")
    lines.set_markersize(5)
    return lines,

#animate_cont function to animate the line in 3D Mesh
def animate_cont(i,line,data):
    line.set_data(data[0:2, :i])
    line.set_marker("o")
    line.set_markersize(5)
    return line,

#this function draws the actual 3D Mesh and calls other helping functions
def Gen_3DMesh_Plot(x_mesh,y_mesh,z_mesh,data):
    fig = plt.figure(figsize=(7,7))
    print(' ')
    ax = plt.axes(projection='3d')
    plt.xlabel('theta[0]')
    print(" ")
    plt.ylabel('theta[1]')
    ax.set_zlabel('Cost Function J(theta)')
    print(" ")
    ax.plot_surface(x_mesh, y_mesh, z_mesh, alpha = 0.8, color='y')
    
    lines, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1],c='r',markersize=0.5)
    print('data.shape[1]',data.shape[1])
    anim1 = animation.FuncAnimation(fig, animate_line, frames=data.shape[1], fargs=(lines,data),interval=200)
    plt.show()
    plt.close()
    return anim1

#this function draws the actual Contour and calls other helping functions
def Gen_Contour_Plot(x_mesh,y_mesh,z_mesh,data):
    fig = plt.figure(figsize=(7,7))
    print('contour plot starts here')
    ax = plt.gca()
    plt.xlabel('theta[0]')
    plt.ylabel('theta[1]')
    print('labels are set')
    plt.contour(x_mesh, y_mesh, z_mesh)
    
    line, = ax.plot(data[0, 0:1], data[1, 0:1],c='r',markersize=0.5)
    print("animation started")
    anim2 = animation.FuncAnimation(fig, animate_cont, frames=data.shape[1], fargs=(line,data),interval=200)
    plt.show()
    plt.close()
    return anim2
#this function draws the error function
def draw_cost_functio(cost_list):
    plt.ylabel('J(theta)')
    plt.xlabel('Iterations')
    plt.plot(cost_list)
    plt.show()
    plt.close()


# In[10]:


def Main_func():
    X,Y,theta,m = read_data()
    #learning rate can be changed here
    learning_rate = .1
    #gradient descent algo is called here
    final_theta,final_cost,iterations,cost_list,data = gradient_descent(X,Y,m,theta,learning_rate)
    print('Final Theta :',final_theta)
    print('Final Cost :',final_cost)
    print('No. of iterations :',iterations)
    print(data[0, 0:1], data[1, 0:1], data[2, 0:1])
    #function to call decision boundary
    plot_decision_boundary(X,Y,final_theta)
    #function to call cost function
    draw_cost_functio(cost_list)
    #function to create data for 3D Mesh and Contour
    x_mesh,y_mesh,z_mesh = Gen_Data_3dMesh(X,Y,m,final_theta)
    #function to create 3D Mesh
    Gen_3DMesh_Plot(x_mesh,y_mesh,z_mesh,data)
   #function to create Contour 
    Gen_Contour_Plot(x_mesh,y_mesh,z_mesh,data)

#------------main function
Main_func()
