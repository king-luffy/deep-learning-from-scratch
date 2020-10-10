# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
        print("W1 :%s,W2:%s,W3:%s" %
              (network['W1'].shape, network['W2'].shape, network['W3'].shape))
    return network


# network : 模型
# x 输入参数
def predict(network, x):
    # 权重
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    # 偏差
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 第一层神经网络
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    # 第二层神经网络
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    # 第三层神经网络
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 最も確率の高い要素のインデックスを取得
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))