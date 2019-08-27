#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Chapter7'))
	print(os.getcwd())
except:
	pass
# ## 7.6 畳み込みニューラルネットワークの実践
# ### 7.6.3 CNNのコード

get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from layer import ConvLayer,PoolingLayer,MiddleLayer,OutputLayer


# -- 手書き文字データセットの読み込み --
digits_data = datasets.load_digits()
input_data = digits_data.data
correct = digits_data.target
n_data = len(correct)

print("n_data" , n_data)
print("input_data" , input_data[0].reshape(8,8))


# -- 入力データの標準化 --
ave_input = np.average(input_data)
std_input = np.std(input_data)
input_data = (input_data - ave_input) / std_input

print("input_data" , input_data[0].reshape(8,8))

# -- 正解をone-hot表現に --
correct_data = np.zeros((n_data, 10))
for i in range(n_data):
    correct_data[i, correct[i]] = 1.0

# -- 訓練データとテストデータ --
index = np.arange(n_data)                # 0から１７９7までの順列
index_train = index[index%3 != 0]
index_test = index[index%3 == 0]

print("index_train" , index_train[:10], "個数", len(index_train))
print("index_test" , index_test[:10], "個数", len(index_test))

input_train = input_data[index_train, :]  # 訓練画像 8×8
correct_train = correct_data[index_train, :]  # 訓練 正解(one-hot表現)
input_test = input_data[index_test, :]  # テスト画像 8×8
correct_test = correct_data[index_test, :]  # テスト 正解(one-hot表現)

print("correct_train\n" , correct_train[:3])

n_train = input_train.shape[0]  # 訓練データのサンプル数
n_test = input_test.shape[0]  # テストデータのサンプル数

print("input_train.shape" , input_train.shape)  # (1198, 64)

# -- 各設定値 --
img_h = 8  # 入力画像の高さ
img_w = 8  # 入力画像の幅
img_ch = 1  # 入力画像のチャンネル数

wb_width = 0.1  # 重みとバイアスの広がり具合
eta = 0.01  # 学習係数
epoch = 1
batch_size = 8
interval = 1  # 経過の表示間隔
n_sample = 7  # 誤差計測のサンプル数
n_flt = 6      # n_flt:フィルタ数
flt_h = 3 # flt_h:フィルタ高さ
flt_w = 3 # flt_w:フィルタ幅
# stride:ストライド幅, pad:パディング幅


# -- 各層の初期化 --
cl_1 = ConvLayer(img_ch, img_h, img_w, n_flt, flt_h, flt_w, 1, 1)
pl_1 = PoolingLayer(cl_1.y_ch, cl_1.y_h, cl_1.y_w, 2, 0)

n_fc_in = pl_1.y_ch * pl_1.y_h * pl_1.y_w
ml_1 = MiddleLayer(n_fc_in, 10)
ol_1 = OutputLayer(10, 10)

# -- 順伝播 --
def forward_propagation(x):
    n_bt = x.shape[0]
    
    print("forward_propagation()")
    print("  x.shape", x.shape)
    images = x.reshape(n_bt, img_ch, img_h, img_w)
    print("  x.reshape", images.shape)

    cl_1.forward(images)
    pl_1.forward(cl_1.y)

    print("  pl_1.y.shape", pl_1.y.shape, "バッチ数,フィルタ数,img縦,img横")
    fullyConnected_input = pl_1.y.reshape(n_bt, -1)
    print("  fc_input.shape", fullyConnected_input.shape)

    ml_1.forward(fullyConnected_input)
    ol_1.forward(ml_1.y)

# -- 逆伝播 --
def backpropagation(t):
    n_bt = t.shape[0]

    ol_1.backward(t)
    ml_1.backward(ol_1.grad_x)

    grad_img = ml_1.grad_x.reshape(n_bt, pl_1.y_ch, pl_1.y_h, pl_1.y_w)
    pl_1.backward(grad_img)
    cl_1.backward(pl_1.grad_x)

# -- 重みとバイアスの更新 --
def uppdate_wb():
    cl_1.update(eta)
    ml_1.update(eta)
    ol_1.update(eta)

# -- 誤差を計算 --
def get_error(t, batch_size):
    print("get_error() 誤差計測サンプル数 × one-hot表現数")
    print("  t.shape", t.shape, " ol_1.y", ol_1.y.shape)
    tmp = t * np.log(ol_1.y + 1e-7)
    print("  tmp.shape", tmp.shape)
    print("  tmp", tmp)
    print("  -np.sum(tmp)", -np.sum(tmp))
    return -np.sum(tmp) / batch_size # 交差エントロピー誤差

# -- サンプルを順伝播 --
def forward_sample(input_dt, correct_dt, n_sample):
    index_rand = np.arange(len(correct_dt))
    np.random.shuffle(index_rand)
    index_rand = index_rand[:n_sample]  # シャッフルデータをサンプル数の上限を切る
    
    input_rand = input_dt[index_rand, :]
    correct_rand = correct_dt[index_rand, :]
    forward_propagation(input_rand)
    return input_rand, correct_rand  # 入力データ、正解データのランダムサンプリング

# -- 誤差の記録用 --
train_error_x = []
train_error_y = []
test_error_x = []
test_error_y = []

# -- 学習と経過の記録 --
n_batch = n_train // batch_size  # 訓練データのサンプル数1198 // バッチサイズ8

print("n_batch" , n_batch)  # バッチ実行回数 149回

n_batch = 1

for i in range(epoch):

    # -- 誤差の計測 --
    x, t = forward_sample(input_train, correct_train, n_sample)
    error_train = get_error(t, n_sample)

    x, t = forward_sample(input_test, correct_test, n_sample)
    error_test = get_error(t, n_sample)
    
    # -- 誤差の記録 --
    train_error_x.append(i)
    train_error_y.append(error_train)
    test_error_x.append(i)
    test_error_y.append(error_test)

    # -- 経過の表示 --
    if i%interval == 0:
        print("Epoch:" + str(i) + "/" + str(epoch),
              "Error_train:" + str(error_train),
              "Error_test:" + str(error_test))

    # -- 学習 --
    index_rand = np.arange(n_train)
    np.random.shuffle(index_rand)
    
    for j in range(n_batch):
        # 各ミニバッチ実行
        mb_index = index_rand[j * batch_size : (j+1) * batch_size]
        
        # 訓練画像 8×8,訓練正解をバッチサイズ分取得
        x = input_train[mb_index, :]
        t = correct_train[mb_index, :]

        forward_propagation(x)
        backpropagation(t)
        uppdate_wb()

# -- 誤差の記録をグラフ表示 --
plt.figure(figsize=(8,5),dpi=100)
plt.plot(train_error_x, train_error_y, label="Train")
plt.plot(test_error_x, test_error_y, label="Test")
plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Error")

plt.show()

# -- 正解率の測定 --
x, t = forward_sample(input_train, correct_train, n_train)
count_train = np.sum(np.argmax(ol_1.y, axis=1) == np.argmax(t, axis=1))

x, t = forward_sample(input_test, correct_test, n_test)
count_test = np.sum(np.argmax(ol_1.y, axis=1) == np.argmax(t, axis=1))

print("Accuracy Train:", str(count_train/n_train*100) + "%\n" +
      "Accuracy Test:", str(count_test/n_test*100) + "%")

samples = input_test[:5]
forward_propagation(samples)
print(ol_1.y)
print(correct_test[:5])

