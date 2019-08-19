# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import matplotlib.pyplot as plt
import numpy as np
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'Chapter4'))
    print(os.getcwd())
except:
    pass
# %% [markdown]
# ### 4.5.7 ニューラルネットワーク（分類）

# %%
get_ipython().run_line_magic('matplotlib', 'inline')


# x、y座標
X = np.arange(-1.0, 1.0, 0.1)  # 要素数は20個
Y = np.arange(-1.0, 1.0, 0.1)

# 重み
w_im = np.array([[1.0, 2.0],
                 [2.0, 3.0]])  # 中間層 2x2の行列
w_mo = np.array([[-1.0, 1.0],
                 [1.0, -1.0]])  # 出力層 2x2の行列

# バイアス
b_im = np.array([0.3, -0.3])  # 中間層
b_mo = np.array([0.4, 0.1])  # 出力層

# 中間層


def middle_layer(x, w, b):
    u = np.dot(x, w) + b
    return 1/(1+np.exp(-u))  # シグモイド関数

# 出力層


def output_layer(x, w, b):
    u = np.dot(x, w) + b
    return np.exp(u)/np.sum(np.exp(u))  # ソフトマックス関数


# 分類結果を格納するリスト
x_1 = []
y_1 = []
x_2 = []
y_2 = []

# グリッドの各マスでニューラルネットワークの演算
for i in range(20):
    for j in range(20):

        # 順伝播
        inp = np.array([X[i], Y[j]])
        mid = middle_layer(inp, w_im, b_im)
        out = output_layer(mid, w_mo, b_mo)

        # 確率の大小を比較し、分類する
        if out[0] > out[1]:
            x_1.append(X[i])
            y_1.append(Y[j])
        else:
            x_2.append(X[i])
            y_2.append(Y[j])

# 散布図の表示
plt.scatter(x_1, y_1, marker="+")
plt.scatter(x_2, y_2, marker="o")
plt.show()


# %%
