# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'Chapter3'))
    print(os.getcwd())
except:
    pass
# %% [markdown]
# ## 3.1 数学の記号について
# %% [markdown]
# ### 3.1.1 シグマ（$\Sigma$）による総和の表記

# %%
import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(np.sum(a))


# %%

a = np.array([2, 1, 2, 1, 2])
b = np.array([1, 2, 1, 2, 1])

print(np.sum(a*b))

# %% [markdown]
# ### 3.1.2 ネイピア数$e$

# %%


def get_exp(x):
    return np.exp(x)


print(get_exp(1))

# %% [markdown]
# ### 3.1.3 自然対数$\log$

# %%


def get_log(x):
    return np.log(x)


print(get_log(1))


# %%
