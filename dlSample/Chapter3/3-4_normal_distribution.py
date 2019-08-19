# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import matplotlib.pyplot as plt
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'Chapter3'))
    print(os.getcwd())
except:
    pass
# %% [markdown]
# ## 3.4 正規分布

# %%
import numpy as np

a = np.array([1, 2, 3, 4, 5])

print("平均値:", np.average(a))
print("標準偏差:", np.std(a))


# %%
get_ipython().run_line_magic('matplotlib', 'inline')


x = np.linspace(-4, 4)
y = 1/np.sqrt(2*np.pi)*np.exp(-x*x/2)

plt.plot(x, y)
plt.tick_params(labelbottom=False, labelleft=False,
                labelright=False, labeltop=False, color="white")
plt.show()


# %%
get_ipython().run_line_magic('matplotlib', 'inline')


# 正規分布に従う乱数を生成 平均50、標準偏差10、10000個
x = np.random.normal(50, 10, 10000)

# ヒストグラム
plt.hist(x, bins=50)  # 50は棒の数
plt.show()


# %%
print(np.average(x))
print(np.std(x))


# %%
