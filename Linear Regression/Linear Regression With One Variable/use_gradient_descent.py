import numpy as np
import matplotlib.pyplot as plt   
from sklearn.linear_model import LinearRegression


def cost_function(x):
    m = A.shape[0]
    return (0.5/m) * np.linalg.norm(A.dot(x) - b, 2) ** 2 

# Tinh dao ham rieng
def grad(x):
    m = A.shape[0]
    return (1/m) * A.T.dot(A.dot(x) - b) 

def gradient_descent(x, learning_rate, iteration):
    x_list = [x]  
    cost_list = []
    for i in range(iteration):
        x_new = x_list[-1] - learning_rate * grad(x_list[-1]) 
        x_list.append(x_new)
        cost_list.append(cost_function(x_new))
    return x_list, cost_list


#* random data
A = [2,5,7,9,11,16,19,23,22,29,29,35,37,40,46]
b = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

#* Convert A from list to numpy array
A = np.array([A]).T
b = np.array([b]).T

#* visualize data 
plt.scatter(A, b, color='green')

#* create one vector 
one = np.ones((A.shape[0], 1), dtype=np.int8)

#* Combine one with A
A = np.concatenate((one, A), axis=1)

#* =============== Draw line use sklearn =============== 
def use_sklearn(x, y):
    reg = LinearRegression()
    reg.fit(x, y)
    x0 = np.linspace(0, 46, 2)
    y0 = reg.intercept_ + reg.coef_[0][1] * x0
    plt.plot(x0, y0, label="use sklearn")
use_sklearn(A, b)

#* =============== Draw line use gradient descent =============== 

# Random hsg bat ky
x_init = np.array([[1, 1]]).T 
learning_rate = 0.001
iteration = 40

def use_gradient_descent(x_init, learning_rate, iteration):
    x_list, _ = gradient_descent(x_init, learning_rate, iteration)

    # draw line
    x0 = np.linspace(0, 46, 2)
    y0 = x_list[-1][0][0] + x_list[-1][1][0]  * x0

    # plot
    plt.plot(x0, y0, color='r', label='use gradient descent')
    plt.title("Gradient descent for Linear Regression")
    plt.xlabel("x value")  
    plt.ylabel("y value")
    plt.grid()


use_gradient_descent(x_init, learning_rate, iteration)
plt.legend(loc='lower right')
plt.show()

def cost_function_plot(iteration):
    # plot cost 
    _, cost_list = gradient_descent(x_init, learning_rate, iteration)
    
    loop = np.arange(1, iteration+1)
    plt.bar(loop, cost_list)
    plt.xlabel("Interation")
    plt.ylabel("Cost func after each iteration")
    plt.title("Cost function")
cost_function_plot(iteration)
plt.show()
