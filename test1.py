import numpy as np

lst = [["a", "b", "c"]
      ,[3, 1, 2]
      ,[3, 1, 2]
#      ,[3, 1, 2]
      ]

print('np.chooseを使って、置換群を作用させる')
print(lst)

def permutation(arg0, *args):
    for arg in args:
        arg0 = np.choose(np.array(arg) - 1 , arg0)
    return arg0

a = permutation(*lst)

print(a)