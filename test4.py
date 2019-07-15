import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

x = np.linspace(0, 1)
y_x = x
y_logx = (-1) * np.log(x + 1e-7)
cross_entropy = lambda x : x * (-1) * np.log(x + 1e-7)

y_cross_entropy = cross_entropy(x)

y_integrate = integrate.quad(cross_entropy, 0, 1)

# 軸のラベル
plt.xlabel("x value")
plt.ylabel("y value")

# グラフのタイトル
plt.title("cross_entropy")

plt.xlim(0,1)
plt.ylim(0,1)

# プロット 凡例と線のスタイルを指定
plt.plot(x, y_x, label="x", linestyle=":")
plt.plot(x, y_logx, label="-logx", linestyle="--")
plt.plot(x, y_cross_entropy, label="cross_entropy", linestyle="-")
#plt.plot(x, y_integrate, label="integrate", linestyle="-.")
plt.legend()  # 凡例を表示

plt.show()
