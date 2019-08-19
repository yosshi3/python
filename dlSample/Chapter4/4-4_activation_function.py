# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'Chapter4'))
    print(os.getcwd())
except:
    pass
# %% [markdown]
# ## 4.4 活性化関数
# %% [markdown]
# ### 4.4.1 ステップ関数

# %%
get_ipython().run_line_magic('matplotlib', 'inline')


def step_function(x):
    return np.where(x <= 0, 0, 1)


x = np.linspace(-5, 5)
y = step_function(x)

plt.plot(x, y)
plt.show()

# %% [markdown]
# ### 4.4.2 シグモイド関数

# %%
get_ipython().run_line_magic('matplotlib', 'inline')


def sigmoid_function(x):
    return 1/(1+np.exp(-x))


x = np.linspace(-5, 5)
y = sigmoid_function(x)

plt.plot(x, y)
plt.show()

# %% [markdown]
# ### 4.4.3 tanh

# %%
get_ipython().run_line_magic('matplotlib', 'inline')


def tanh_function(x):
    return np.tanh(x)


x = np.linspace(-5, 5)
y = tanh_function(x)

plt.plot(x, y)
plt.show()

# %% [markdown]
# ### 4.4.4 ReLU

# %%
get_ipython().run_line_magic('matplotlib', 'inline')


def relu_function(x):
    return np.where(x <= 0, 0, x)


x = np.linspace(-5, 5)
y = relu_function(x)

plt.plot(x, y)
plt.show()

# %% [markdown]
# ### 4.4.5 Leaky ReLU

# %%
get_ipython().run_line_magic('matplotlib', 'inline')


def leaky_relu_function(x):
    return np.where(x <= 0, 0.01*x, x)


x = np.linspace(-5, 5)
y = leaky_relu_function(x)

plt.plot(x, y)
plt.show()

# %% [markdown]
# ### 4.4.6 恒等関数

# %%
get_ipython().run_line_magic('matplotlib', 'inline')


x = np.linspace(-5, 5)
y = x

plt.plot(x, y)
plt.show()

# %% [markdown]
# ### 4.4.7 ソフトマックス関数

# %%


def softmax_function(x):
    return np.exp(x)/np.sum(np.exp(x))  # ソフトマックス関数


y = softmax_function(np.array([1, 2, 3]))
print(y)


# %%
