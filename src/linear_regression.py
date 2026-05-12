# 线性回归的从零开始实现
# 包括数据流水线、模型、损失函数和小批量随机梯度下降优化器

import random

import torch
from d2l import torch as d2l
from torch.nn.modules import loss


# 根据带有噪声的线性模型构造一个人造数据集。
# 使用线性模型参数 w = [2, -3.4]，b = 4.2 和噪声项生成数据集及其标签
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声"""
    # 生成 X: 均值为 0，标准差为 1 的随机数，形状为 (num_examples, len(w))
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 生成 y: X 与 w 的矩阵乘法，加上 b，再加上噪声
    y = torch.matmul(X, w) + b
    # 生成噪声: 均值为 0，标准差为 0.01 的随机数，形状与 y 相同
    y += torch.normal(0, 0.01, y.shape)
    # 返回 X 和 y，y 被 reshape 为列向量
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)


# 定义数据迭代器
# 参数：批量大小、特征矩阵、标签向量
# 返回：大小为 batch_size 的特征和标签的批量
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 打乱索引
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i : min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, "\n", y)
    break


# 定义初始化模型参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# 定义模型
def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b


# 定义损失函数
def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 定义优化算法：小批量随机梯度下降
def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 训练模型
# 迭代次数、学习率、批量大小
num_epochs = 3
lr = 0.03
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        # X 和 y 的小批量损失
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f"epoch {epoch + 1}, loss {float(train_l.mean()):f}")


# 比较真实参数和训练学来的参数，来评估训练的成功程度
print(f"w的估计误差：{true_w - w.reshape(true_w.shape)}")
print(f"b的估计误差：{true_b - b}")
