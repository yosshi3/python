# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import matplotlib.pylab as plt
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'Chapter5'))
    print(os.getcwd())
except:
    pass
# %% [markdown]
# ## 5.4 損失関数
# %% [markdown]
# ### 5.4.1 二乗和誤差

# %%
import numpy as np


def square_sum(y, t):
    return 1.0/2.0 * np.sum(np.square(y - t))


err = square_sum(np.array([1, 2, 3]), np.array([2, 3, 4]))
print(err)

# %% [markdown]
# ### 5.4.2 交差エントロピー誤差

# %%
get_ipython().run_line_magic('matplotlib', 'inline')


x = np.linspace(1.0e-03, 1)
y = - np.log(x)

plt.plot(x, y)
plt.show()


# %%


def cross_entropy(y, t):  # 出力、正解
    return - np.sum(t * np.log(y + 1e-7))


err = cross_entropy(np.array([0.9, 0.1, 0.1]), np.array([1, 0, 0]))
print(err)


# %%
