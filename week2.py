import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

# PAM-8 符号与比特相互转换 (使用格雷码)
# 标准 PAM8 电平映射到格雷码 (字符串表示方便理解)
symbol_to_bit_map = {
    -7: "000",
    -5: "001",
    -3: "011",
    -1: "010",
    1: "110",
    3: "111",
    5: "101",
    7: "100",
}


def calculate_ber(y_true_symbols, y_pred_symbols):
    """计算比特误码率（BER）"""
    bit_errors = 0
    total_bits = len(y_true_symbols) * 3  # 每个符号 3 个比特

    for true_sym, pred_sym in zip(y_true_symbols, y_pred_symbols):
        true_bits = symbol_to_bit_map[true_sym]
        pred_bits = symbol_to_bit_map[pred_sym]
        # 逐位对比，不同则错误数 + 1
        bit_errors += sum(1 for a, b in zip(true_bits, pred_bits) if a != b)

    return bit_errors / total_bits


# 定义全连接神经网络 (MLP) [cite: 6, 9]
class MLP(nn.Module):
    def __init__(self, input_size=15):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # 输出连续值用于回归判决
        )

    def forward(self, x):
        return self.net(x)


# 定义 3层 1维卷积神经网络 (1D CNN) [cite: 13, 15]
class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        # 假设输入特征维度为 (Batch, Channels=1, Sequence_Length=15)
        self.conv_layers = nn.Sequential(
            # 第一层卷积 [cite: 15]
            # 输出长度: 15-3+1 = 13
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3),
            nn.ReLU(),
            # 第二层卷积 [cite: 15]
            # 输出长度: 13-3+1 = 11
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            # 第三层卷积 [cite: 15]
            # 输出长度: 11-3+1 = 9
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 9, 1)  # 9 是输出长度

    def forward(self, x):
        # 注意：Conv1d 需要输入维度为 [N, C, L]，我们需要在 forward 里增加通道维度
        x = x.unsqueeze(1)  # 把 (Batch, 15) 变成 (Batch, 1, 15)
        x = self.conv_layers(x)
        x = self.flatten(x)
        return self.fc(x)


# 定义 2维卷积神经网络 (2D CNN)
class CNN2D(nn.Module):
    def __init__(self):
        super(CNN2D, self).__init__()
        # 假设我们将输入形状调整为 (Batch_size, Channels=1, Height=1, Width=15)
        self.conv_layers = nn.Sequential(
            # 第一层 2D 卷积，卷积核尺寸为 (1, 3)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 3)),
            nn.ReLU(),
            # 第二层 2D 卷积
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3)),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        # 经过两次核宽为 3 的无 Padding 卷积，Width 变为 15 - 2 - 2 = 11
        self.fc = nn.Linear(32 * 1 * 11, 1)

    def forward(self, x):
        # 注意：Conv2d 需要输入维度为 [N, C, H, W]
        # x 初始是 (Batch, 15)，连续增加两个维度变成 (Batch, 1, 1, 15)
        x = x.unsqueeze(1).unsqueeze(1)
        x = self.conv_layers(x)
        x = self.flatten(x)
        return self.fc(x)


# 训练主流程与 Tensorboard 集成 [cite: 20, 21]
def train_and_evaluate(
    model, X_train, Y_train, X_test, Y_test, epochs=200, lr=0.005, run_name="MLP_Run"
):
    # 初始化 TensorBoard Writer
    writer = SummaryWriter(f"runs/{run_name}")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    pam8_levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()
        # 将训练 Loss 写入TensorBoard
        writer.add_scalar("Loss/train", loss.item(), epoch)
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
    # 测试集评估计算 BER
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test).numpy().flatten()
        test_trues = Y_test.numpy().flatten()

        # 1. 硬判决：拉回到最近的 PAM8 符号
        pred_symbols = np.array(
            [pam8_levels[np.argmin(np.abs(pam8_levels - p))] for p in test_preds]
        )
        # 2. 计算 BER [cite: 4, 10, 15]
        ber = calculate_ber(test_trues, pred_symbols)
        # 将最终 BER 写入 TensorBoard
        writer.add_scalar("Metrics/Test_BER", ber, 0)
        print(f"[{run_name}] 最终测试集 BER: {ber:.6f}")

    writer.close()
