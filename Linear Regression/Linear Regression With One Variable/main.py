import numpy as np
import matplotlib.pyplot as plt   


#* random data
A = [2,5,7,9,11,16,19,23,22,29,29,35,37,40,46]
b = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

#* Visualize data
plt.scatter(A, b, color='green')


#* Change row vector to column vector
A = np.array([A]).T
b = np.array([b]).T


#* Create vector one
one = np.ones((A.shape[0], 1), dtype=np.int8)

#* Concatenate one with A
A = np.concatenate((A, one), axis=1)


#* Use formula 
x = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(b)
print(x)

# Draw line 
x0 = np.linspace(0, 46, 2)
y0 = x0*x[0][0] + x[1][0]
plt.plot(x0, y0, 'red')
plt.title('Linear Regression With One Variable')
plt.xlabel('x value') 
plt.ylabel('y value')
plt.grid()
plt.show()





