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
# ### 4.5.1 単一のニューロンを実装

# %%
get_ipython().run_line_magic('matplotlib', 'inline')


# x、y座標
X = np.arange(-1.0, 1.0, 0.2)  # 要素数は10個
Y = np.arange(-1.0, 1.0, 0.2)

# 出力を格納する10x10のグリッド
Z = np.zeros((10, 10))

# x、y座標の入力の重み
w_x = 2.5
w_y = 3.0

# バイアス
bias = 0.1

# グリッドの各マスでニューロンの演算
for i in range(10):
    for j in range(10):

        # 入力と重みの積の総和 + バイアス
        u = X[i]*w_x + Y[j]*w_y + bias

        # グリッドに出力を格納
        y = 1/(1+np.exp(-u))  # シグモイド関数
        Z[j][i] = y

# グリッドの表示
plt.imshow(Z, "gray", vmin=0.0, vmax=1.0)
plt.colorbar()
plt.show()


# %%
