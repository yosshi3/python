import numpy as np
from functools import reduce

lst = [ 
        ["a", "b", "c"]
        ,[3, 1, 2]
        ,[3, 1, 2]
#        ,[3, 1, 2]
        ]

e = reduce(lambda x, y: np.choose(np.array(y) - 1, x), lst)

print(e)
