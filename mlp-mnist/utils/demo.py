import torch
import torch.nn as nn
import numpy as np


# 定义 MLP 类
class MLP(nn.Module):
	def __init__(self, dim_in, dim_hidden, dim_out):
		super(MLP, self).__init__()
		self.layer_input = nn.Linear(dim_in, dim_hidden)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout()
		self.layer_hidden = nn.Linear(dim_hidden, dim_out)
	
	def forward(self, x):
		x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
		x = self.layer_input(x)
		x = self.dropout(x)
		x = self.relu(x)
		x = self.layer_hidden(x)
		return x


# 初始化 MLP 模型
model = MLP(dim_in=784, dim_hidden=128, dim_out=10)

# 获取第二层权重矩阵
weight_fc2 = model.layer_hidden.weight.detach().numpy()

# 示例数据
x_new = torch.randn(10, 128)  # 新数据，与第二层的输入大小相匹配


# 对权重参数进行增量 SVD
def incremental_svd(param, x_new):
	U, Sigma, Vt = np.linalg.svd(param, full_matrices=False)
	print(Vt.T.shape)#128*10
	
	p = np.dot(x_new, Vt.T)
	new_param = np.hstack((param, p))
	return new_param


# 对第二层权重矩阵进行增量 SVD
updated_weight_fc2 = incremental_svd(weight_fc2, x_new)

# 更新模型的第二层权重矩阵
model.layer_hidden.weight.data = torch.from_numpy(updated_weight_fc2)

# 打印更新后的模型参数
print(model)























