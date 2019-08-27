import numpy as np
from im2col_col2im import im2col,col2im

wb_width = 0.1  # 重みとバイアスの広がり具合

# -- 畳み込み層 --
class ConvLayer:

    # n_bt:バッチサイズ, x_ch:入力チャンネル数, x_h:入力画像高さ, x_w:入力画像幅
    # n_flt:フィルタ数, flt_h:フィルタ高さ, flt_w:フィルタ幅
    # stride:ストライド幅, pad:パディング幅
    # y_ch:出力チャンネル数, y_h:出力高さ, y_w:出力幅

    # x_ch:入力チャンネル数, x_h:入力画像高さ, x_w:入力画像幅
    # n_flt:フィルタ数, flt_h:フィルタ高さ, flt_w:フィルタ幅
    # stride:ストライド幅, pad:パディング幅
    def __init__(self, x_ch, x_h, x_w, n_flt, flt_h, flt_w, stride, pad):

        # パラメータをまとめる
        self.params = (x_ch, x_h, x_w, n_flt, flt_h, flt_w, stride, pad)

        # フィルタとバイアスの初期値
        self.w = wb_width * np.random.randn(n_flt, x_ch, flt_h, flt_w)
        # バイアスの初期値（1,フィルタ数）
        self.b = wb_width * np.random.randn(1, n_flt)

        # 出力画像のサイズ
        self.y_ch = n_flt  # 出力チャンネル数 ★★★★★★
        self.y_h = (x_h - flt_h + 2*pad) // stride + 1  # 出力高さ
        self.y_w = (x_w - flt_w + 2*pad) // stride + 1  # 出力幅

        # AdaGrad用
        self.h_w = np.zeros((n_flt, x_ch, flt_h, flt_w)) + 1e-8
        self.h_b = np.zeros((1, n_flt)) + 1e-8

    def forward(self, x):
        n_bt = x.shape[0]
        x_ch, x_h, x_w, n_flt, flt_h, flt_w, stride, pad = self.params
        y_ch, y_h, y_w = self.y_ch, self.y_h, self.y_w

        # 入力画像とフィルタを行列に変換
        # im2colsで行列形式に変換
        self.cols = im2col(x, flt_h, flt_w, y_h, y_w, stride, pad)
        # フィルタを行列形式に変換
        self.w_col = self.w.reshape(n_flt, x_ch * flt_h * flt_w)

        # 出力の計算: 行列積、バイアスの加算、活性化関数
        # フィルタとフィルタ該当画像を掛けて、フィルタ後の結果にchannelを含めて
        # self.w_col = n_flt  ,  x_ch * flt_h * flt_w
        # self.cols = x_ch * flt_h * flt_w  ,  n_bt * y_h * y_w
        # np.dot() = n_flt  ,  n_bt * y_h * y_w
        # np.dot().T = n_bt * y_h * y_w  ,  n_flt
        # self.b = 1  ,  n_flt
        # u = n_bt * y_h * y_w  ,  n_flt
        u = np.dot(self.w_col, self.cols).T + self.b
        # u = n_bt, y_h, y_w, y_ch reshape()
        # u = n_bt, y_ch, y_h, y_w transpose()
        self.u = u.reshape(n_bt, y_h, y_w, y_ch).transpose(0, 3, 1, 2)
        # y = n_bt, y_ch, y_h, y_w
        print("forward()")
        print("  self.u", self.u.shape)
        self.y = np.where(self.u <= 0, 0, self.u)
        print("  self.y", self.y.shape)

    def backward(self, grad_y):
        n_bt = grad_y.shape[0]
        x_ch, x_h, x_w, n_flt, flt_h, flt_w, stride, pad = self.params
        y_ch, y_h, y_w = self.y_ch, self.y_h, self.y_w

        # delta
        delta = grad_y * np.where(self.u <= 0, 0, 1)
        delta = delta.transpose(0,2,3,1).reshape(n_bt * y_h * y_w, y_ch)

        # フィルタとバイアスの勾配
        grad_w = np.dot(self.cols, delta)
        self.grad_w = grad_w.T.reshape(n_flt, x_ch, flt_h, flt_w)
        self.grad_b = np.sum(delta, axis=0)

        # 入力の勾配
        grad_cols = np.dot(delta, self.w_col)
        x_shape = (n_bt, x_ch, x_h, x_w)
        self.grad_x = col2im(grad_cols.T, x_shape, flt_h, flt_w, y_h, y_w, stride, pad)

    def update(self, eta):
        self.h_w += self.grad_w * self.grad_w
        self.w -= eta / np.sqrt(self.h_w) * self.grad_w

        self.h_b += self.grad_b * self.grad_b
        self.b -= eta / np.sqrt(self.h_b) * self.grad_b
        print("update()")
        print("  self.grad_b.shape", self.grad_b.shape)
        print("  self.grad_b", self.grad_b)
        print("  self.b.shape", self.b.shape)
        print("  self.b", self.b)

# -- プーリング層 --
class PoolingLayer:

    # n_bt:バッチサイズ, x_ch:入力チャンネル数, x_h:入力画像高さ, x_w:入力画像幅
    # pool:プーリング領域のサイズ, pad:パディング幅
    # y_ch:出力チャンネル数, y_h:出力高さ, y_w:出力幅

    def __init__(self, x_ch, x_h, x_w, pool, pad):

        # パラメータをまとめる
        self.params = (x_ch, x_h, x_w, pool, pad)

        # 出力画像のサイズ
        self.y_ch = x_ch  # 出力チャンネル数
        self.y_h = x_h//pool if x_h%pool==0 else x_h//pool+1  # 出力高さ
        self.y_w = x_w//pool if x_w%pool==0 else x_w//pool+1  # 出力幅

    def forward(self, x):
        n_bt = x.shape[0]
        x_ch, x_h, x_w, pool, pad = self.params
        y_ch, y_h, y_w = self.y_ch, self.y_h, self.y_w

        # 入力画像を行列に変換
        cols = im2col(x, pool, pool, y_h, y_w, pool, pad)
        cols = cols.T.reshape(n_bt * y_h * y_w * x_ch, pool * pool)

        # 出力の計算: Maxプーリング
        y = np.max(cols, axis=1)
        self.y = y.reshape(n_bt, y_h, y_w, x_ch).transpose(0, 3, 1, 2)

        # 最大値のインデックスを保存
        self.max_index = np.argmax(cols, axis=1)
        print("forward()")
        print("  y", y.shape)
        print("  self.max_index", self.max_index.shape)

    def backward(self, grad_y):
        n_bt = grad_y.shape[0]
        x_ch, x_h, x_w, pool, pad = self.params
        y_ch, y_h, y_w = self.y_ch, self.y_h, self.y_w

        print("backward()")
        # 出力の勾配の軸を入れ替え
        grad_y = grad_y.transpose(0, 2, 3, 1)
        print("  grad_y", grad_y.shape)

        # 行列を作成し、各列の最大値であった要素にのみ出力の勾配を入れる
        grad_cols = np.zeros((pool * pool, grad_y.size))
        print("  grad_cols", grad_cols.shape)
        tmp1 = self.max_index.reshape(-1)
        print("  tmp1", tmp1.shape)
        tmp2 = np.arange(grad_y.size)
        print("  tmp2", tmp2.shape)
        grad_cols[tmp1, tmp2] = grad_y.reshape(-1)
        print("  grad_cols", grad_cols.shape)
        grad_cols = grad_cols.reshape(pool, pool, n_bt, y_h, y_w, y_ch)
        print("  grad_cols", grad_cols.shape)
        grad_cols = grad_cols.transpose(5,0,1,2,3,4)
        print("  grad_cols", grad_cols.shape)
        grad_cols = grad_cols.reshape( y_ch*pool*pool, n_bt*y_h*y_w)
        print("  grad_cols", grad_cols.shape)

        # 入力の勾配
        x_shape = (n_bt, x_ch, x_h, x_w)
        self.grad_x = col2im(grad_cols, x_shape, pool, pool, y_h, y_w, pool, pad)


# -- 全結合層の継承元 --
class BaseLayer:
    def __init__(self, n_upper, n):
        self.w = wb_width * np.random.randn(n_upper, n)
        self.b = wb_width * np.random.randn(n)

        self.h_w = np.zeros(( n_upper, n)) + 1e-8
        self.h_b = np.zeros(n) + 1e-8

    def update(self, eta):
        self.h_w += self.grad_w * self.grad_w
        self.w -= eta / np.sqrt(self.h_w) * self.grad_w

        self.h_b += self.grad_b * self.grad_b
        self.b -= eta / np.sqrt(self.h_b) * self.grad_b

# -- 全結合 中間層 --
class MiddleLayer(BaseLayer):
    def forward(self, x):
        self.x = x
        self.u = np.dot(x, self.w) + self.b
        self.y = np.where(self.u <= 0, 0, self.u)

    def backward(self, grad_y):
        delta = grad_y * np.where(self.u <= 0, 0, 1)

        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)

        self.grad_x = np.dot(delta, self.w.T)

# -- 全結合 出力層 --
class OutputLayer(BaseLayer):
    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = np.exp(u)/np.sum(np.exp(u), axis=1).reshape(-1, 1)

    def backward(self, t):
        delta = self.y - t

        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)

        self.grad_x = np.dot(delta, self.w.T)



# if __name__ == '__main__':
