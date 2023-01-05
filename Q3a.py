import numpy as np
from numpy import array

x=array([[1,2,3],[4,5,6]])
y= array([[4,5,6],[3,2,1]])
z= array([[3,2,1],[1,2,3]])

#find sqrt(x)

x2 = np.multiply(x,x)
y2 = np.multiply(2,y)
z2 = np.multiply(z,z)
z3 = np.multiply(z2,z)

res1 = np.add(x2,y2)
res2 = np.subtract(res1,z3)
print("\nProblem : X^2 + 2y - z^3:\n")
print("Solution:\n")

# print("\nx^2")
# print(x2)
# print("\n2y")
# print(y2)
# print("\nz^3")
# print(z3)
# print("\nFinal Result")
print(res2)