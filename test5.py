import numpy as np

#x + 2y + 3z + 4w = 4 6 6 4 =20
#2x + 3y + 4z + w = 8 9 8 1 = 26
#3x + 4y + 1z +2w = 12 12 2 2 =28
#1x + 1y + 1z + 1w = 1 2 3 4 = 10

A = np.array([[1,2,3,4],
              [2,3,4,1],
              [3,4,1,2],
              [1,1,1,1]])

B = np.array([20,26,28,10])

C = B.T

inv_A = np.linalg.inv(A)

print(inv_A @ B)
