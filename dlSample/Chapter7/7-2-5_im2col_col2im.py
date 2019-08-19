# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'Chapter7'))
    print(os.getcwd())
except:
    pass

import numpy as np

# ### 7.2.6 col2imの実装

def col2im(cols, img_shape, flt_h, flt_w, out_h, out_w, stride, pad):

    n_bt, n_ch, img_h, img_w = img_shape

    cols = cols.reshape(n_ch, flt_h, flt_w, n_bt, out_h,
                        out_w, ).transpose(3, 0, 1, 2, 4, 5)
    images = np.zeros((n_bt, n_ch, img_h+2*pad+stride-1, img_w+2*pad+stride-1))

    for h in range(flt_h):
        h_lim = h + stride*out_h
        for w in range(flt_w):
            w_lim = w + stride*out_w
            images[:, :, h:h_lim:stride,
                   w:w_lim:stride] += cols[:, :, h, w, :, :]

    return images[:, :, pad:img_h+pad, pad:img_w+pad]


cols = np.ones((4, 4))
img_shape = (1, 1, 3, 3)
images = col2im(cols, img_shape, 2, 2, 2, 2, 1, 0)
print("images = col2im(cols, img_shape, 2, 2, 2, 2, 1, 0)\n" + str(images))


