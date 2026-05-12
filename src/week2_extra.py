import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter

from utils import calculate_ber


# 任务 3 & 4: 构建优化的 Dataset 和 DataLoader
# 自定义 VLC 数据集，采用“延迟切片”策略，极大节省显存。
# 不提前构建 Nx15 的特征矩阵，而是在每次读取数据时动态截取
class VLCDataset(Dataset):
    def __init__(self, tx_data, rx_padded, indices, window_size=15):
        # 转换为 Tensor
        self.tx_data = torch.tensor(tx_data, dtype=torch.float32)
        self.rx_padded = torch.tensor(rx_padded, dtype=torch.float32)
        self.indices = indices
        self.window_size = window_size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        # 获取真实的全局索引
        idx = self.indices[i]

        # 动态获取滑动窗口数据作为输入 X (长度15)
        x = self.rx_padded[idx : idx + self.window_size]
        # 获取对应的发送标签 Y (长度1)
        y = self.tx_data[idx].view(1)

        return x, y


def main():
    # 提取数据
    tx_data = sio.loadmat("./data/exp15_paras.mat")["originPAM"].flatten()
    rx_data = sio.loadmat("./data/exp15_CHAN1_2859.mat")["pamRecv"].flatten()

    # 数据预处理 (仅在接收信号两端补零，一维数组，占用内存极小)
    window_size = 15
    pad_size = window_size // 2
    rx_padded = np.pad(rx_data, (pad_size, pad_size), "constant")

    # 划分数据集索引 (前10000训练，后5000测试)
    train_indices = np.arange(0, 10000)
    test_indices = np.arange(10000, 15000)

    # 实例化 Dataset
    train_dataset = VLCDataset(tx_data, rx_padded, train_indices, window_size)
    test_dataset = VLCDataset(tx_data, rx_padded, test_indices, window_size)

    # 实例化 DataLoader (构建小批量训练流)
    batch_size = 256
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 模型、损失、优化器与 TensorBoard
    model = nn.Linear(15, 1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    writer = SummaryWriter("runs/Week2_Extra")

    # 训练循环
    epochs = 50
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        avg_loss = epoch_loss / len(train_dataset)
        # 补充：写入 TensorBoard
        writer.add_scalar("Loss/train", avg_loss, epoch)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_loss:.4f}")

    # 补充：评估与测试环节 (Validation/Testing Loop)
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        # 遍历测试集 DataLoader，收集所有的预测值和真实值
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            all_preds.extend(outputs.numpy().flatten())
            all_targets.extend(batch_y.numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    pam8_levels = np.unique(tx_data)

    # 判决逻辑：找最近的标准 PAM8 电平
    pred_decisions = np.array(
        [pam8_levels[np.argmin(np.abs(pam8_levels - p))] for p in all_preds]
    )

    # 计算指标
    errors = np.sum(pred_decisions != all_targets)
    total_test_symbols = len(all_targets)
    ser = errors / total_test_symbols
    ber = calculate_ber(all_targets, pred_decisions)

    print("\n" + "=" * 30)
    print(f"测试集样本总数: {total_test_symbols}")
    print(f"错误判决个数: {errors}")
    print(f"PAM8 误符号率 (SER): {ser:.6f}")
    print(f"PAM8 比特误码率 (BER): {ber:.6f}")

    # 将最终评估指标写入 TensorBoard
    writer.add_scalar("Metrics/Test_SER", ser, epochs)
    writer.add_scalar("Metrics/Test_BER", ber, epochs)
    writer.close()


if __name__ == "__main__":
    main()
