import numpy as np

a = np.arange(24).reshape(2,3,4).transpose(0,2,1)

print('np.arange(24), a.reshape, a.transposeを使って2,4,3行列作成')
print(a)

a = a + 1

print('ブロードキャストで1足す')
print(a)

a = a[1,:,:]

print('スライシングで、1次元削除')
print(a)

a[2:,1:] -= np.ones(a[2:,1:].shape, dtype = "int8")
#a[2:,1:] -= np.ones_like(a[2:,1:])

print('3行目以降、2列目以降の2,2行列に対してnp.onesを使って1引く')
print(a)

