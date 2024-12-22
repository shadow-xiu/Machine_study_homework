import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 设置matplotlib字体，以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# Step 1: 数据加载与预处理
class TEProcessDataset(Dataset):
    def __init__(self, file_paths, seq_len=10):
        self.data = []
        self.seq_len = seq_len

        # 加载所有数据文件
        min_columns = None
        for file_path in file_paths:
            df = pd.read_csv(file_path, sep=r'\s+', header=None)  # 使用 sep='\s+'
            if min_columns is None:
                min_columns = df.shape[1]  # 记录最小列数
            else:
                min_columns = min(min_columns, df.shape[1])
            self.data.append(df)

        # 裁剪列数，使所有文件列数一致
        self.data = [df.iloc[:, :min_columns].values for df in self.data]

        # 拼接所有数据
        self.data = np.concatenate(self.data, axis=0)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len, :-1]  # 输入特征
        y = self.data[idx + self.seq_len, -1]  # 目标值
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def generate_file_paths(base_dir, file_prefix, num_files, ext):
    return [os.path.join(base_dir, f"{file_prefix}{str(i).zfill(2)}{ext}") for i in range(num_files)]

# 基本的文件路径
base_test_dir = r"D:\桌面\pythonProject2\data\TE_data\test"
base_train_dir = r"D:\桌面\pythonProject2\data\TE_data\train"

# 生成测试集文件路径
test_files = generate_file_paths(base_test_dir, "d", 22, "_te.dat")

# 生成训练集文件路径
train_files = generate_file_paths(base_train_dir, "d", 22, ".dat")

# 创建数据集
train_dataset = TEProcessDataset(train_files)
test_dataset = TEProcessDataset(test_files)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Step 2: 构建RNN模型
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

input_size = train_dataset.data.shape[1] - 1  # 特征数
hidden_size = 64
num_layers = 2
output_size = 1
model = RNNModel(input_size, hidden_size, num_layers, output_size)

# Step 3: 模型训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 400
train_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    print(f'Epoch(当前训练的进度) [{epoch+1}/{num_epochs}], 损失: {epoch_loss:.4f}')

# Step 4: 模型评估
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        predictions.extend(outputs.cpu().numpy())
        actuals.extend(targets.cpu().numpy())

# 计算评估指标
mae = mean_absolute_error(actuals, predictions)
mse = mean_squared_error(actuals, predictions)
rmse = np.sqrt(mse)

print(f'平均绝对误差MAE: {mae:.4f}')
print(f'均方误差MSE: {mse:.4f}')
print(f'均方根误差RMSE: {rmse:.4f}')

# Step 5: 可视化训练损失
plt.plot(train_losses, label='训练损失')
plt.xlabel('周期')
plt.ylabel('损失')
plt.title('训练损失曲线')
plt.legend()
plt.show()