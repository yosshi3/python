import numpy as np

mat = np.arange(0,16)
print("mat", mat)

mat = mat.reshape(4,4)
print("mat(4,4)\n", mat)

mat = mat.reshape(2,2,2,2)
print("mat(2,2,2,2)\n[0,0]\n", mat[0,0], 
      "\n[0,1]\n", mat[0,1], 
      "\n[1,0]\n", mat[1,0], 
      "\n[1,1]\n", mat[1,1])

