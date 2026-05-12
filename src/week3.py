import time

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from utils import calculate_ber


# ==========================================
# 1. 封装 EXNN 神经网络模型
# ==========================================
class EXNN(nn.Module):
    def __init__(self, channels=32):  # 这里补充了默认的通道数(可按需修改)
        super(EXNN, self).__init__()
        kernels = 3
        # 1D 卷积层组成的网络
        self.net = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=kernels, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=kernels, dilation=1, padding=1),
            nn.ReLU(),
            # dilation=2, padding=2 保证输入输出序列长度一致
            nn.Conv1d(channels, channels, kernel_size=kernels, dilation=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=kernels, dilation=4, padding=4),
            nn.ReLU(),
            nn.Conv1d(channels, 1, kernel_size=kernels, stride=1, padding=1),
        )

    def forward(self, x):
        # x 的输入形状要求: (Batch_Size, 1_Channel, 2048_Length)
        return self.net(x)


# ==========================================
# 2. 构造 Dataset (处理数据切分和步长)
# ==========================================
class Week3Dataset(Dataset):
    def __init__(self, rx_data, tx_data, start_idx, end_idx, window_size=2048, step=16):
        self.rx_data = torch.tensor(rx_data, dtype=torch.float32)
        self.tx_data = torch.tensor(tx_data, dtype=torch.float32)
        self.window_size = window_size

        # 核心：根据 start, end 和步长 step 生成起始索引
        self.indices = np.arange(start_idx, end_idx - window_size + 1, step)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]

        # 取 2048 个点，并增加一个通道维度 unsqueeze(0)
        # 结果形状变为: (1, 2048)
        x = self.rx_data[idx : idx + self.window_size].unsqueeze(0)
        y = self.tx_data[idx : idx + self.window_size].unsqueeze(0)
        return x, y


# ==========================================
# 3. 训练主流程
# ==========================================
def main():
    # 设备检测逻辑
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"当前使用设备: {device}")

    # ================= 模拟数据加载 =================
    # TODO: 请替换为你真实的读取不同工况(mA, Vpp)的 mat 文件代码
    # tx_data = sio.loadmat("工况A_TX.mat")["xxx"].flatten()
    # rx_data = sio.loadmat("工况A_RX.mat")["xxx"].flatten()
    # 这里为了代码能直接跑，暂时随机生成长度为 15000 的数据模拟
    total_length = 15000
    tx_data = sio.loadmat("./data/exp15_paras.mat")["originPAM"].flatten()
    rx_data = sio.loadmat("./data/exp15_CHAN1_2859.mat")["pamRecv"].flatten()

    window_size = 2048

    # ================= 按照要求划分数据集 =================
    # 训练集: 1至15000-2048*3 (索引 0 到 15000-6144 = 8856)，步长 16
    train_dataset = Week3Dataset(
        rx_data, tx_data, start_idx=0, end_idx=8856, window_size=window_size, step=16
    )

    # 验证集: 15000-3*2048 至 15000-2*2048 (索引 8856 到 10904)
    # 验证集步长文档没写，一般可以等于 window_size(不重叠) 或更小，这里设为 2048
    val_dataset = Week3Dataset(
        rx_data,
        tx_data,
        start_idx=8856,
        end_idx=10904,
        window_size=window_size,
        step=2048,
    )

    # 测试集: 15000-2*2048 至 15000 (索引 10904 到 15000)
    test_dataset = Week3Dataset(
        rx_data,
        tx_data,
        start_idx=10904,
        end_idx=15000,
        window_size=window_size,
        step=2048,
    )

    # ================= 按照要求设置超参数 =================
    batch_size = 256
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = EXNN(channels=32).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)  # 学习率 5e-4

    epochs = 150  # 训练轮数 150

    # 重置显存统计
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(f"训练集 batch 数量: {len(train_loader)}")

    # ================= 训练循环 =================
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        # 记录训练起始时间
        start_time = time.time()

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        end_time = time.time()
        avg_loss = epoch_loss / len(train_dataset)

        # 计算每一条数据的平均处理时间
        time_per_batch = (end_time - start_time) / len(train_loader)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] | Loss: {avg_loss:.4f} | 耗时: {end_time - start_time:.2f}s | 单个Batch平均耗时: {time_per_batch:.4f}s"
            )

            # (建议3.a) 打印显存占用
            if device.type == "cuda":
                max_mem = torch.cuda.max_memory_allocated() / (1024**2)
                print(f"   => 显存峰值占用: {max_mem:.2f} MB")

    # ==========================================
    # 4. 测试集评估环节 (计算 SER 与 BER)
    # ==========================================
    print("\n" + "=" * 40)
    print("开始在测试集上进行评估...")

    # 将模型设置为评估模式（冻结 Dropout 和 BatchNorm，如果有的话）
    model.eval()
    all_preds = []
    all_targets = []

    # 禁用梯度计算，节省显存并加速
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            # 前向传播得到预测序列
            outputs = model(batch_x)
            all_preds.append(outputs)
            # batch_y 原本就在 CPU (或未显式 to(device))，稳妥起见也加上 cpu()
            all_targets.append(batch_y.to(device))

    # 将列表中的批次张量拼接成一个完整的一维/多维大张量
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    # 定义 PAM-8 标准理想电平
    pam8_levels_tensor = torch.tensor(
        [-7, -5, -3, -1, 1, 3, 5, 7], device=device
    ).float()

    print("正在进行电平判决并计算误码率 (由于数据量大，可能需要几秒钟)...")

    # 判决逻辑优化：使用 NumPy 广播机制快速计算距离并寻找最近电平
    # 距离矩阵形状: (N, 8)
    # 在 GPU 上计算距离矩阵
    # 使用广播机制: (N, 1) - (1, 8) -> (N, 8)
    dist = (all_preds.view(-1, 1) - pam8_levels_tensor.view(1, -1)).abs()
    # 找到距离最近的电平索引，并映射回真实电平
    pred_decisions = pam8_levels_tensor[torch.argmin(dist, dim=1)]

    # 1. 计算误符号率 (SER)
    total_test_symbols = all_targets.numel()  # 获取总元素个数
    errors = (pred_decisions != all_targets.view(-1)).sum().item()
    ser = errors / total_test_symbols

    # 2. 计算误比特率 (BER)
    # 注意：我们的 utils.py 里的 calculate_ber 主要是基于 Python/NumPy 写的
    # 所以在最后一步，我们需要把数据从 GPU 拿回 CPU，再转成 NumPy 传给它
    all_targets_cpu = all_targets.view(-1).cpu().numpy()
    pred_decisions_cpu = pred_decisions.cpu().numpy()
    ber = calculate_ber(all_targets_cpu, pred_decisions_cpu)

    # 打印最终结果
    print("\n" + "🚀 评估结果 🚀".center(36))
    print("-" * 40)
    print(f"测试集样本总点数 : {total_test_symbols}")
    print(f"错误判决符号个数 : {errors}")
    print(f"PAM8 误符号率(SER): {ser:.6f}")
    print(f"PAM8 误比特率(BER): {ber:.6f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
