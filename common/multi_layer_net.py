# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


class MultiLayerNet:
    """全链接的多层神经网络

     参数
     ----------
     input_size：输入大小（MNIST为784）
     hidden_size_list：隐藏层中神经元数量的列表（例如[100，100，100]）
     output_size：输出大小（MNIST为10）
     激活：“relu”或“ Sigmoid”
     weight_init_std：指定权重的标准偏差（例如0.01）
         如果指定了“relu”或“he”，则设置“He的初始值”
         如果指定“sigmoid”或“xavier”，则设置“ Xavier的初始值”
     weight_decay_lambda：权重衰减强度（L2规范）

    """
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        # 权重初始化
        self.__init_weight(weight_init_std)

        # 层生成
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
            self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """权重初始值设定

         参数
         ----------
         weight_init_std：指定权重的标准偏差（例如0.01）
             如果指定了“relu”或“he”，则设置“He的初始值”
             如果指定“sigmoid”或“xavier”，则设置“Xavier的初始值”
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # 使用ReLU时的建议初始值
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # 使用sigmoid时的建议初始值

            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """
        求损失函数

         参量
         ----------
         x：输入数据
         t：训练标签

         返回
         -------
         损失函数值
        """
        y = self.predict(x)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """求梯度（数值微分）

         参数
         ----------
         x：输入数据
         t：老师标签

         退货
         -------
         每层渐变的字典变量
             渐变['W1']，渐变['W2']，...是每一层的权重
             渐变['b1']，渐变['b2']，...是每一层的偏差
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """找到梯度（误差反向传播方法）

         参数
         ----------
         x：输入数据
         t：老师标签

         退货
         -------
         每层渐变的字典变量
             渐变['W1']，渐变['W2']，...是每一层的权重
             渐变['b1']，渐变['b2']，...是每一层的偏差
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads
