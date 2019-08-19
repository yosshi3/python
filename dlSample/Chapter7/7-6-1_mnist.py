# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
from sklearn import datasets
import matplotlib.pyplot as plt
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'Chapter7'))
    print(os.getcwd())
except:
    pass
# %% [markdown]
# ## 7.6 畳み込みニューラルネットワークの実践
# %% [markdown]
# ### 7.6.1 使用するデータセット

# %%
get_ipython().run_line_magic('matplotlib', 'inline')


digits = datasets.load_digits()
print("digits.data.shape:" + str(digits.data.shape))

# 0番目のデータの内容表示
print("digits.data[0].reshape(8,8):" + str(digits.data[0].reshape(8, 8)))
plt.imshow(digits.data[0].reshape(8, 8), cmap="gray")
plt.show()


# %%
print(digits.target.shape)
print(len(digits.target))
print(digits.target[:50].reshape(5, 10))


# %%
get_ipython().run_line_magic('matplotlib', 'inline')


digits = datasets.load_digits()

fig = plt.figure(figsize=(10, 10), dpi=100)  # 表示サイズを大きくする

for i in range(50):
    ax = fig.add_subplot(5, 10, i + 1)    # 5×10で表示する
    ax.tick_params(labelbottom="off", bottom="off")
    ax.tick_params(labelleft="off", left="off")
    plt.imshow(digits.data[i].reshape(8, 8), cmap="gray")
    plt.title(digits.target[i])

plt.show()
