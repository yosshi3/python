# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'Chapter7'))
    print(os.getcwd())
except:
    pass

import numpy as np



def im2col(images, flt_h, flt_w, out_h, out_w, stride, pad):

    print("images.shape = " + str(images.shape))
    n_bt, n_ch, img_h, img_w = images.shape

    pad_width = [(0, 0), (0, 0), (pad, pad), (pad, pad)]
    img_pad = np.pad(images, pad_width, "constant")
    print("img_pad = " + str(img_pad))
    
    cols = np.zeros((n_bt, n_ch, flt_h, flt_w, out_h, out_w), dtype=np.int16)

    for h in range(flt_h):
        h_lim = h + stride*out_h
        for w in range(flt_w):
            w_lim = w + stride*out_w
            cols[:, :, h, w, :, :] = img_pad[: , : , h:h_lim:stride, w:w_lim:stride]
            print("img_pad[] = " + str(img_pad[: , : , h:h_lim:stride, w:w_lim:stride]))
            print("cols = " + str(cols))

    cols = cols.transpose(1, 2, 3, 0, 4, 5)
    cols = cols.reshape(n_ch * flt_h * flt_w , n_bt * out_h * out_w)
    return cols

img = np.array([[[[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]]]])
#cols = im2col(img, 2, 2, 3, 3, 1, 0)
#print("cols = im2col(img, 2, 2, 3, 3, 1, 0)\n" + str(cols))

cols = im2col(img, 2, 2, 5, 5, 1, 1)
print("cols = im2col(img, 2, 2, 3, 3, 1, 1)\n" + str(cols))


