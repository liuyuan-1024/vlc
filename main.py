import time

import torch

# 1. 准备设备
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🔥 正在使用设备: {device}")

# 2. 创建两个巨大的张量（模拟真实深度学习计算）
size = 4000
x = torch.randn(size, size, device=device)
y = torch.randn(size, size, device=device)

# 3. 计时开始
start_time = time.time()

# 4. 执行矩阵乘法（这是显卡最擅长的事）
result = torch.matmul(x, y)

# 5. 等待 GPU 完成计算（同步）
if device == "mps":
    torch.mps.synchronize()

end_time = time.time()
print(f"✅ 成功！{size}x{size} 矩阵乘法耗时: {end_time - start_time:.4f} 秒")
