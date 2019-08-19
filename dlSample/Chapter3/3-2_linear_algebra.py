#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Chapter3'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# ## 3.2 線形代数
#%% [markdown]
# ### 3.2.1 スカラー

#%%
a = 1
b = 1.2
c = -0.25
d = 1.2e5

#%% [markdown]
# ### 3.2.2 ベクトル

#%%
import numpy as np

a = np.array([1, 2, 3])  # 縦ベクトルとして扱う
b = np.array([-2.3, 0.25, -1.2, 1.8, 0.41])

#%% [markdown]
# ### 3.2.3 行列

#%%
import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6]])  # 2x3の行列
b = np.array([[0.21, 0.14],
              [-1.3, 0.81],
              [0.12, -2.1]])  # 3x2の行列

#%% [markdown]
# ### 3.2.4 テンソル

#%%
import numpy as np

a = np.array([[[0, 1, 2, 3],
               [2, 3, 4, 5],
               [4, 5, 6, 7]],

              [[1, 2, 3, 4],
               [3, 4, 5, 6],
               [5, 6, 7, 8]]])  # (2, 3, 4)の3階のテンソル


#%%
# 1階のテンソル（ベクトル）
b = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])

# 2階のテンソル（行列）
b = b.reshape(4, 6)
print(b)


#%%
# 3階のテンソル
b = b.reshape(2, 3, 4)
print(b)


#%%
# 4階のテンソル
b = b.reshape(2, 2, 3, 2)
print(b)


#%%
c = np.array([[[1,2,3,4],
               [2,0,0,0],
               [3,0,0,0]],

              [[2,0,0,0],
               [0,0,0,0],
               [0,0,0,0]]])  # (2, 3, 4)の3階のテンソル


#%%
c = c.transpose(0, 2, 1)  # (2, 4, 3)
print(c)


#%%
c = c.transpose(2, 0, 1)  # (3, 2, 4)
print(c)


#%%
c = c.transpose(1, 0, 2)  # (2, 3, 4)
print(c)

#%% [markdown]
# ### 3.2.5 スカラーと行列の積

#%%
import numpy as np

c = 2
a = np.array([[0, 1, 2],
              [3, 4, 5],
              [6, 7, 8]])

print(c*a)

#%% [markdown]
# ### 3.2.6 要素ごとの積

#%%
import numpy as np

a = np.array([[0, 1, 2],
              [3, 4, 5],
              [6, 7, 8]])
b = np.array([[0, 1, 2],
              [2, 0, 1],
              [1, 2, 0]])

print(a*b)


#%%
print(a+b)  # 足し算


#%%
print(a-b)  # 引き算


#%%
print(a/(b+1))  # 割り算 1を足すのはゼロ除算対策


#%%
print(a%(b+1))  # 余り 1を足すのはゼロ除算対策

#%% [markdown]
# ### 3.2.7 行列積

#%%
import numpy as np

a = np.array([[0, 1, 2],
              [1, 2, 3]])
b = np.array([[2, 1],
              [2, 1],
              [2, 1]])

print(np.dot(a, b))


#%%
import numpy as np

a = np.array([1, 2, 3])  # 行数1の行列として扱われる
b = np.array([[1, 2],
              [1, 2],
              [1, 2]])

print(np.dot(a, b))

#%% [markdown]
# ### 3.2.8 行列の転置

#%%
import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6]])
print(a.T)  # 転置


#%%
import numpy as np

a = np.array([[0, 1, 2],
              [1, 2, 3]])  # 2x3
b = np.array([[0, 1, 2],
              [1, 2, 3]])  # 2x3 このままでは行列積ができない

# print(np.dot(a, b))  # エラー
print(np.dot(a, b.T))  # 転置により行列積が可能に


#%%



