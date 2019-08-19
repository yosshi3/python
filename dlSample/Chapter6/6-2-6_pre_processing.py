# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'Chapter6'))
    print(os.getcwd())
except:
    pass
# %% [markdown]
# ### 6.2.6 データの前処理
# %% [markdown]
# #### 正規化

# %%
import numpy as np


def normalize(x):
    x_max = np.max(x)
    x_min = np.min(x)
    return (x - x_min) / (x_max - x_min)


print(normalize(np.array([1, 2, 3, 4])))

# %% [markdown]
# #### 標準化

# %%


def standardize(x):
    ave = np.average(x)
    std = np.std(x)
    return (x - ave) / std


print(standardize(np.array([1, 2, 3, 4])))


# %%
