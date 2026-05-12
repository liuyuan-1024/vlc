import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim

# 导入 TensorBoard
from torch.utils.tensorboard.writer import SummaryWriter

# 提取数据并展平为一维数组
tx_data = sio.loadmat("exp15_paras.mat")["originPAM"].flatten()
rx_data = sio.loadmat("exp15_CHAN1_2859.mat")["pamRecv"].flatten()

print(f"tx 长度: {len(tx_data)}, rx 长度: {len(rx_data)}")

# 数据预处理 (构建滑动窗口)
window_size = 15
pad_size = window_size // 2  # 中心位置为 7，两边各补 7 个零

# 在接收信号两端补零，保证滑动窗口提取后总数量仍为 15000
Rx_padded = np.pad(rx_data, (pad_size, pad_size), "constant")

# 构建输入特征矩阵 X (15000 x 15) 和标签 Y (15000 x 1)
X = np.zeros((len(tx_data), window_size))
for i in range(len(tx_data)):
    X[i] = Rx_padded[i : i + window_size]
Y = tx_data.reshape(-1, 1)

# 转换为 PyTorch 的 Tensor 格式
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# 按要求划分：前10000个为训练集，后5000个为测试集
X_train = X_tensor[:10000]
Y_train = Y_tensor[:10000]
X_test = X_tensor[10000:15000]
Y_test = Y_tensor[10000:15000]


# 构建线性回归模型
class LinearCompensator(nn.Module):
    def __init__(self):
        super(LinearCompensator, self).__init__()
        # 输入维度 15 (15个Rx)，输出维度 1 (预测中心的1个Tx)
        self.linear = nn.Linear(15, 1)

    def forward(self, x):
        return self.linear(x)


model = LinearCompensator()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 2. 初始化 TensorBoard Writer，指定保存路径
writer = SummaryWriter("runs/Week1_Linear")

# 训练模型
epochs = 500  # 迭代次数
for epoch in range(epochs):
    model.train()  # 设置为训练模式
    optimizer.zero_grad()  # 梯度清零

    outputs = model(X_train)  # 前向传播
    loss = criterion(outputs, Y_train)  # 计算损失

    loss.backward()  # 反向传播求梯度
    optimizer.step()  # 更新权重

    # 3. 将每一个 epoch 的 Loss 写入 TensorBoard
    writer.add_scalar("Loss/train", loss.item(), epoch)

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")


# 十进制转二进制的基础函数
def decimal_to_binary(decimal_num, bit_length=3):
    """将十进制转换为固定长度的二进制字符串"""
    if decimal_num < 0:
        raise ValueError("仅支持非负整数")
    return format(decimal_num, f"0{bit_length}b")


# 动态映射：PAM符号 -> 格雷码比特串
def pam_to_gray_bits(symbol):
    """将 PAM-8 符号 (-7 到 7) 转换为对应的 3bit 格雷码字符串"""
    # 步骤A: 将符号电平映射为 0~7 的十进制索引 (-7->0, -5->1 ... 7->7)
    index = int((symbol + 7) // 2)

    # 步骤B: 将自然二进制索引转换为格雷码十进制值 (公式: Gray = n XOR (n右移1位))
    gray_dec = index ^ (index >> 1)

    # 步骤C: 使用十进制转二进制函数输出最终比特
    return decimal_to_binary(gray_dec, 3)


# 计算 BER 函数
def calculate_ber(y_true_symbols, y_pred_symbols):
    """计算比特误码率（BER）"""
    bit_errors = 0
    # PAM-8 每个符号 3 个比特
    total_bits = len(y_true_symbols) * 3

    for true_sym, pred_sym in zip(y_true_symbols, y_pred_symbols):
        # 动态生成格雷码比特，不再依赖写死的字典
        true_bits = pam_to_gray_bits(true_sym)
        pred_bits = pam_to_gray_bits(pred_sym)

        # 逐位对比，不同则错误数 + 1
        bit_errors += sum(1 for a, b in zip(true_bits, pred_bits) if a != b)

    return bit_errors / total_bits


# 测试并计算 PAM8 误符号率 (SER) 与 误比特率 (BER)
model.eval()  # 设置为评估模式
with torch.no_grad():
    predictions = model(X_test).numpy().flatten()
    targets = Y_test.numpy().flatten()

    # 提取 PAM8 的标准理想电平值
    pam8_levels = np.unique(tx_data)

    # 判决逻辑：对于每一个预测出的连续值，找到距离它最近的标准 PAM8 电平
    pred_decisions = np.array(
        [pam8_levels[np.argmin(np.abs(pam8_levels - p))] for p in predictions]
    )

    # 计算错误符号数
    errors = np.sum(pred_decisions != targets)
    total_test_symbols = len(targets)
    # 计算误符号率 SER
    ser = errors / total_test_symbols

    # 计算误比特率 BER
    ber = calculate_ber(targets, pred_decisions)

    print("-" * 30)
    print(f"测试集样本总数: {total_test_symbols}")
    print(f"错误判决个数: {errors}")
    print(f"PAM8 误符号率 (SER): {ser:.6f}")
    print(f"PAM8 比特误码率 (BER): {ber:.6f}")

    # 4. 将最终的 SER 和 BER 写入 TensorBoard
    writer.add_scalar("Metrics/Test_SER", ser, 0)
    writer.add_scalar("Metrics/Test_BER", ber, 0)

# 5. 关闭 Writer
writer.close()
