# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'Chapter2'))
    print(os.getcwd())
except:
    pass
# %% [markdown]
# ## 2.4 Numpy
# %% [markdown]
# ### 2.3.1 Numpyのインポート

# %%
import numpy as np

# %% [markdown]
# ### 2.4.2 Numpyの配列

# %%
a = np.array([0, 1, 2, 3, 4, 5])
print(a)


# %%
b = np.array([[0, 1, 2], [3, 4, 5]])  # リストのリストを渡す
print(b)


# %%
c = np.array([[[0, 1, 2], [3, 4, 5]], [[5, 4, 3], [2, 1, 0]]])
print(c)


# %%
print(np.shape(c))
print(np.size(c))

['a1_a1','a1_a2'],[a2_a1	a2_a2
['a1_b1','a1_b2]		a2_b1	a2_b2
				
'b1_a1','b1_a2		b2_a1	b2_a2
'b1_b1','b1_b2		b2_b1	b2_b2




# %%
d = [[1, 2], [3, 4], [5, 6]]  # (3, 2)
print(len(d))
print(len(np.array(d)))

# %% [markdown]
# ### 2.4.3 配列を生成する様々な関数

# %%
print(np.zeros(10))
print(np.ones(10))
print(np.random.rand(10))


# %%
print(np.zeros((2, 3)))
print(np.ones((2, 3)))


# %%
print(np.arange(0, 1, 0.1))


# %%
print(np.arange(10))


# %%
print(np.linspace(0, 1, 11))


# %%
print(np.linspace(0, 1))

# %% [markdown]
# ### 2.4.4 reshapeによる形状の変換

# %%
a = np.array([0, 1, 2, 3, 4, 5, 6, 7])    # 配列の作成
b = a.reshape(2, 4)                       # (2, 4)の2次元配列に変換
print(b)


# %%
c = b.reshape(2, 2, 2)  # (2, 2, 2)の3次元配列に変換
print(c)


# %%
d = c.reshape(4, 2)  # (4, 2)の3次元配列に変換
print(d)


# %%
e = d.reshape(-1)  # 1次元配列に変換
print(e)


# %%
f = e.reshape(2, -1)
print(f)

# %% [markdown]
# ### 2.4.5 配列の演算

# %%
a = np.array([0, 1, 2, 3, 4, 5]).reshape(2, 3)
print(a)


# %%
print(a + 3)  # 各要素に3を足す


# %%
print(a * 3)  # 各要素に3をかける


# %%
b = np.array([5, 4, 3, 2, 1, 0]).reshape(2, 3)
print(b)


# %%
print(a + b)


# %%
print(a * b)

# %% [markdown]
# ### 2.4.6 ブロードキャスト

# %%
a = np.array([[1, 1],
              [1, 1]])  # 2次元配列
b = np.array([1, 2])  # 1次元配列


# %%
print(a + b)


# %%
c = np.array([[1],
              [2]])  # 2次元配列


# %%
print(a + c)

# %% [markdown]
# ### 2.4.7 要素へのアクセス

# %%
a = np.array([0, 1, 2, 3, 4, 5])
print(a[2])


# %%
a[2] = 9
print(a)


# %%
b = np.array([[0, 1, 2],
              [3, 4, 5]])
print(b[1, 2])  # b[1][2]と同じ


# %%
b[1, 2] = 9
print(b)


# %%
c = np.array([[0, 1, 2],
              [3, 4, 5],
              [6, 7, 8]])
print(c[1])  # インデックスが1つだけ


# %%
c[1] = np.array([9, 10, 11])  # 要素を配列で置き換え
print(c)


# %%
d = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(d[d % 2 == 0])  # [ ]内に条件を指定


# %%
e = np.zeros((3, 3))  # 2次元配列、要素は全て0
f = np.array([8, 9])  # 置き換える値

e[np.array([0, 2]), np.array([0, 1])] = f  # 2つの配列でインデッックスを指定
print(e)

# %% [markdown]
# ### 2.4.8 スライシング

# %%
a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(a[2:8])  # スライシング


# %%
print(a[2:8:2])


# %%
print(a[:])


# %%
b = np.array([[0, 1, 2],
              [3, 4, 5],
              [6, 7, 8]])
print(b[0:2, 0:2])  # 各次元の範囲を指定


# %%
b[0:2, 0:2] = np.array([[9, 9],
                        [9, 9]])  # 左上の領域を置き換え
print(b)


# %%
c = np.zeros(18).reshape(2, 3, 3)
print(c)


# %%
c[0, 0:2, 0:2] = np.ones(4).reshape(2, 2)
print(c)

# %% [markdown]
# ### 2.4.9 軸とtranspose

# %%
a = np.array([[0, 1, 2],
              [3, 4, 5]])
print(a)


# %%
print(a.transpose(1, 0))  # 軸を入れ替え


# %%
print(a.T)  # 転置


# %%
b = np.arange(12).reshape(2, 2, 3)
print(b)


# %%
print(b.transpose(1, 2, 0))

# %% [markdown]
# ### 2.4.10 NumPyの関数

# %%
a = np.array([[0, 1],
              [2, 3]])
print(np.sum(a))


# %%
print(np.sum(a, axis=0))


# %%
print(np.sum(a, axis=1))


# %%
print(np.sum(a, axis=1, keepdims=True))


# %%
print(np.max(a))


# %%
print(np.max(a, axis=0))


# %%
print(np.argmax(a, axis=0))


# %%
print(np.where(a < 2, 9, a))


# %%
