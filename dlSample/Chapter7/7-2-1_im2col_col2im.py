# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'Chapter7'))
    print(os.getcwd())
except:
    pass

# ### 7.2.3 im2colの実装 -シンプルなim2col-

import numpy as np

test = np.arange(1,17).reshape(4,4)
print("np.array(1,17)\n" + str(test))

  #     入力画像、フィルタの高さ、幅、出力画像の高さ、幅
def im2col(image, flt_h, flt_w, out_h, out_w):

    img_h, img_w = image.shape  # 入力画像の高さ、幅

    cols = np.zeros((flt_h * flt_w, out_h * out_w))  # 生成される行列のサイズ

    for h in range(out_h):
        h_lim = h + flt_h
        for w in range(out_w):
            w_lim = w + flt_w
            tmp = img[h:h_lim, w:w_lim]
            tmp2 = tmp.reshape(-1)
            tmp3 = h * out_w+w
            cols[:, tmp3] = tmp2

    return cols


img = np.array(test)
cols = im2col(img, 2, 2, 3, 3) # 入力画像、フィルタの高さ、幅、出力画像の高さ、幅
print("cols = im2col(img, 2, 2, 3, 3)\n" + str(cols))

