#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Chapter2'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# ## 2.2 Anaconda、Jupyter Notebookの導入

#%%
print("Hello World")


#%%
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-np.pi, np.pi)
plt.plot(x, np.cos(x))
plt.plot(x, np.sin(x))
plt.show()


#%%



