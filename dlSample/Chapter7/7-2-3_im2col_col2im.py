# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'Chapter7'))
    print(os.getcwd())
except:
    pass

import numpy as np

# ### 7.2.4 im2colの実装 -im2colの実用化-

def im2col(images, flt_h, flt_w, out_h, out_w):

    print("images.shape = " + str(images.shape))
    n_bt, n_ch, img_h, img_w = images.shape  # バッチサイズ、チャンネル数、入力画像高さ、幅

    cols = np.zeros((n_bt, n_ch, flt_h, flt_w, out_h, out_w))

    for h in range(flt_h):
        h_lim = h + out_h
        for w in range(flt_w):
            w_lim = w + out_w
            cols[:, :, h, w, :, :] = images[:, :, h:h_lim, w:w_lim]

                # バッチ、チャネル、フィルタ縦、フィルタ横、アウト縦、アウト横 から
                # チャネル、フィルタ縦、フィルタ横 × バッチ、アウト縦、アウト横 へ変換
    cols = cols.transpose(1, 2, 3, 0, 4, 5)
    cols = cols.reshape(n_ch * flt_h * flt_w , n_bt * out_h * out_w)
    return cols

img = np.array([[[[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]]]])
cols = im2col(img, 2, 2, 3, 3)
print("cols = im2col(img, 2, 2, 3, 3)\n" + str(cols))

