# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import matplotlib.pyplot as plt
import numpy as np
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'Chapter2'))
    print(os.getcwd())
except:
    pass
# %% [markdown]
# ## 2.5 Matplotlib
# %% [markdown]
# ### 2.5.1 モジュールのインポート

# %%
get_ipython().run_line_magic('matplotlib', 'inline')


# %% [markdown]
# ### 2.5.2 グラフの描画

# %%
x = np.linspace(0, 2*np.pi)  # 0から2πまで
y = np.sin(x)

plt.plot(x, y)
plt.show()

# %% [markdown]
# ### 2.5.3 グラフの装飾

# %%
x = np.linspace(0, 2*np.pi)
y_sin = np.sin(x)
y_cos = np.cos(x)

# 軸のラベル
plt.xlabel("x value")
plt.ylabel("y value")

# グラフのタイトル
plt.title("sin/cos")

# プロット 凡例と線のスタイルを指定
plt.plot(x, y_sin, label="sin")
plt.plot(x, y_cos, label="cos", linestyle="dashed")
plt.legend()  # 凡例を表示

plt.show()

# %% [markdown]
# ### 2.5.4 散布図の表示

# %%
x_1 = np.random.rand(100) - 1.0  # このグループを左に1.0ずらす
y_1 = np.random.rand(100)
x_2 = np.random.rand(100)
y_2 = np.random.rand(100)

plt.scatter(x_1, y_1, marker="+")  # 散布図のプロット
plt.scatter(x_2, y_2, marker="x")

plt.show()

# %% [markdown]
# ### 2.5.5 画像の表示

# %%
img = np.array([[0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15]])

plt.imshow(img, "gray")  # グレースケールで表示
plt.colorbar()   # カラーバーの表示
plt.show()


# %%
img = plt.imread("flower.png")

plt.imshow(img)
plt.show()


# %%
