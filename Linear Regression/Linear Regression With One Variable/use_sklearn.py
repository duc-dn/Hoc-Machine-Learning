import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# * random data
A = [2, 5, 7, 9, 11, 16, 19, 23, 22, 29, 29, 35, 37, 40, 46]
b = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

A = np.array([A]).T
b = np.array([b]).T
# * Visualize data
plt.scatter(A, b, color='green')

# *
reg = LinearRegression()
reg.fit(A, b)

x0 = np.linspace(0, 46, 2)
y0 = reg.intercept_ + reg.coef_[0][0] * x0
print(reg.coef_)
plt.plot(x0, y0)
plt.title('Linear Regression use sklearn')
plt.xlabel('x value')
plt.ylabel('y value')
plt.grid()
plt.show()
