#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Chapter2'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# ## 2.3 Pythonの文法
#%% [markdown]
# ### 2.3.1 変数と型

#%%
a = 123


#%%
a = 123  # 整数型
b = 123.456  # 浮動小数点型
c = "Hello World!"  # 文字列型
d = True  # 論理型
e = [1, 2, 3]  #リスト型


#%%
a = 123
print(type(a))


#%%
a = True; b = False
print(a+b)


#%%
1.2e5   # 1.2x10の5乗 120000
1.2e-5  # 1.2x10の-5乗 0.000012

#%% [markdown]
# ### 2.3.2 演算子

#%%
a = 3; b = 4

c = a + b               # 足し算
print(c)

d = a < b               # 比較（小さいかどうか）
print(d)

e = 3 < 4 and 4 < 5     # 論理和
print(e)


#%%
a = "Hello" + "World"
print(a)

b = [1, 2, 3] + [4, 5, 6]
print(b)

#%% [markdown]
# ### 2.3.3 リスト

#%%
a = [1, 2, 3, 4, 5]     # リストの作成

b = a[2]                # 3番目の要素を取得
print(b)

a.append(6)             # 末尾に要素を追加する
print(a)

a[2] = 7                # 要素の入れ替え
print(a)

#%% [markdown]
# ### 2.3.4 タプル

#%%
a = (1, 2, 3, 4, 5)     # タプルの作成

b = a[2]                # 3番目の要素を取得
print(b)


#%%
print(a + (6, 7, 8, 9, 10))


#%%
a = [1, 2, 3]
a1, a2, a3 = a
print(a1, a2, a3)

b = (4, 5, 6)
b1, b2, b3 = b
print(b1, b2, b3)

#%% [markdown]
# ### 2.3.5 辞書

#%%
a = {"Apple":3, "Pineapple":4}      # 辞書の作成

print(a["Apple"])       # "Apple"のキーを持つ値を取得

a["Pinapple"] = 6       # 要素の入れ替え
print(a["Pinapple"])

a["Melon"] = 3          # 要素の追加
print(a)

#%% [markdown]
# ### 2.3.6 if文

#%%
a = 7
if a < 12:
    print("Good morning!")
elif a < 17:
    print("Good afternoon!")
elif a < 21:
    print("Good evening!")
else:
    print("Good night!")

#%% [markdown]
# ### 2.3.7 for文

#%%
for a in [4, 7, 10]:    # リストを使ったループ
    print(a)

for a in range(3):      # rangeを使ったループ
    print(a)

#%% [markdown]
# ### 2.3.8 while文

#%%
a = 0
while a < 3:
    print(a)
    a += 1

#%% [markdown]
# ### 2.3.9 内包表記

#%%
a = [1, 2, 3, 4, 5, 6, 7]
b = [c*2 for c in a]    # aの要素を2倍して新たなリストを作る
print(b)


#%%
a = [1, 2, 3, 4, 5, 6, 7]
b = [c*2 for c in a if c < 5]
print(b)

#%% [markdown]
# ### 2.3.10 関数

#%%
def add(a, b):          # 関数の定義
    c = a + b
    return c

print(add(3, 4))        # 関数の実行


#%%
def add(a, b=4):        # 第2引数にデフォルト値を設定
    c = a + b
    return c

print(add(3))           # 第2引数は指定しない


#%%
def add(a, b ,c):
    d = a + b + c
    print(d)

e = (1, 2, 3)
add(*e)           # 複数の引数を一度に渡す

#%% [markdown]
# ### 2.3.11 変数のスコープ

#%%
a = 123         # グローバル変数

def showNum():
    b = 456     # ローカル変数
    print(a, b)

showNum()


#%%
a = 123

def setLocal():
    a = 456     # aはローカル変数とみなされる
    print("Local:", a)

setLocal()
print("Global:", a)


#%%
a = 123

def setGlobal():
    global a            # nonlocalでも可
    a = 456
    print("Global:", a)

setGlobal()
print("Global:", a)

#%% [markdown]
# ### 2.3.12 クラス

#%%
class Calc:
    def __init__(self, a):
        self.a = a

    def add(self, b):
        print(self.a + b)

    def multiply(self, b):
        print(self.a * b)


#%%
calc = Calc(3)
calc.add(4)
calc.multiply(4)


#%%
class CalcPlus(Calc):     # Calcを継承
    def subtract(self, b):
        print(self.a - b)

    def divide(self, b):
        print(self.a / b)


#%%
calc_plus = CalcPlus(3)
calc_plus.add(4)
calc_plus.multiply(4)
calc_plus.subtract(4)
calc_plus.divide(4)


#%%



