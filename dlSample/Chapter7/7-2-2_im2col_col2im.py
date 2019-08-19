# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'Chapter7'))
    print(os.getcwd())
except:
    pass

import numpy as np

# ### 7.2.3 im2colの実装 -シンプルなim2col-


def im2col(image, flt_h, flt_w, out_h, out_w):

    img_h, img_w = image.shape
#    cols = np.zeros((flt_h, flt_w, out_h, out_w))
    cols = np.zeros((flt_h * flt_w, out_h, out_w))

    for h in range(flt_h):  # フィルター縦でループ
        h_lim = h + out_h
        for w in range(flt_w):  # フィルター横でループ
            w_lim = w + out_w
#            cols[h, w, :, :] = img[h:h_lim, w:w_lim] # colsのx行目を生成する
            cols[h * (flt_w) + w, :, :] = img[h:h_lim, w:w_lim]  # colsのx行目を生成する
            print("h = " + str(h) + ": w = " + str(w) + "\n")
            print(str(cols))
            print('---')
        print('★★★')
    cols = cols.reshape(flt_h * flt_w, out_h * out_w)

    return cols


img = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]])
cols = im2col(img, 2, 2, 3, 3)
print("cols = im2col(img, 2, 2, 3, 3)\n" + str(cols))
