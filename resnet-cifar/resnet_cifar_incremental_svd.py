import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from Fed import FFD_2, k_svd_2
import pandas as pd
import openpyxl
from tqdm import tqdm
from torch.linalg import svd

class ResidualBlock(nn.Module):
	def __init__(self, inchannel, outchannel, stride=1):
		super(ResidualBlock, self).__init__()
		self.left = nn.Sequential(
			nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
			nn.BatchNorm2d(outchannel),
			nn.ReLU(inplace=True),
			nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(outchannel)
		)
		self.shortcut = nn.Sequential()
		if stride != 1 or inchannel != outchannel:
			self.shortcut = nn.Sequential(
				nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(outchannel)
			)
	
	def forward(self, x):
		out = self.left(x)
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class ResNet(nn.Module):
	def __init__(self, ResidualBlock, num_classes=10):
		super(ResNet, self).__init__()
		self.inchannel = 64
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(),
		)
		self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
		self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
		self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
		self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
		self.fc = nn.Linear(512, num_classes)
	
	def make_layer(self, block, channels, num_blocks, stride):
		strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
		layers = []
		for stride in strides:
			layers.append(block(self.inchannel, channels, stride))
			self.inchannel = channels
		return nn.Sequential(*layers)
	
	def forward(self, x):
		out = self.conv1(x)
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = F.avg_pool2d(out, 4)
		out = out.view(out.size(0), -1)
		logits = self.fc(out)
		probas = F.softmax(logits, dim=1)
		return logits, probas


def ResNet18():
	return ResNet(ResidualBlock)


# train process
print("\nLOAD DATA\n")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128

# 准备数据集并预处理
transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
	transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 训练数据集
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

# 生成一个个batch进行批训练，组成batch的时候顺序打乱取
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
# 测试数据集
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)

##################################################### Using a GPU #####################################################

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print("\nUSING", device)
if cuda:
	num_dev = torch.cuda.current_device()
	print(torch.cuda.get_device_name(num_dev), "\n")
# 模型定义-ResNet
model = ResNet18().to(DEVICE)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
# 选用Adam优化比SGD的效果要好
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
valid_loader = test_loader


def compute_accuracy_and_loss(model, data_loader, device):
	correct_pred, num_examples = 0, 0
	cross_entropy = 0
	for i, (features, targets) in enumerate(data_loader):
		features = features.to(device)
		targets = targets.to(device)
		
		logits, probas = model(features)
		cross_entropy += F.cross_entropy(logits, targets).item()
		_, predicted_labels = torch.max(probas, 1)
		num_examples += targets.size(0)
		correct_pred += (predicted_labels == targets).sum()
	return correct_pred / num_examples * 100, cross_entropy / num_examples


train_acc_list, valid_acc_list = [], []
train_loss_list, valid_loss_list = [], []

# print(w_global.keys())
# print(len(w_global.keys()))

# count = 0

# for weight in w_global.keys():
# 	name = w_global[f'{weight}']
# 	print(name.shape)
# 	num_params = name.numel()
# 	count += num_params
# print(count)


model.train()
w_global = model.state_dict()
# print(w_glob)
conv1_weight = w_global['conv1.0.weight']
# print(conv1_weight.shape)#(64, 3, 3, 3)

start_svd_time = time.time()
# 获取第二层权重矩阵
weight_fc2 = w_global['conv1.0.weight'].detach().cpu().numpy()

# 示例数据
x_new = torch.ones(64, 3, 3, 3).detach().cpu().numpy()  # 新数据
#x_new = x_new.reshape(64,27)


# 对权重参数进行增量 SVD
def incremental_svd(param, x_new):
	U, Sigma, Vt = np.linalg.svd(param, full_matrices=False)
	Vt = Vt.T
	# print(param.shape)#10,3,5,5
	p = np.dot(x_new.reshape(64, 27), Vt.reshape(27, 64))
	#print(f"p shape: {p.shape}")  # 64*64

	param_tran = param.reshape(-1, param.shape[0])  # 27*64
	#print(f"param tran shape:{param_tran.shape}")

	new_param = np.hstack((param_tran.reshape(64, 27), p))
	#print(f"new param shape:{new_param.shape}")

	return new_param



# 对第二层权重矩阵进行增量 SVD
updated_weight_fc2 = incremental_svd(weight_fc2, x_new)
#print(updated_weight_fc2.shape)#64*91

#end_svd_time = time.time()

# 更新模型的第二层权重矩阵
updated_weight = updated_weight_fc2[:, :27].reshape(64,3,3,3)
#print(updated_weight.shape)

update_conv_weight = torch.from_numpy(updated_weight)
update_conv_weight_1 = update_conv_weight
#print(update_conv_weight_1.shape)
w_global['conv1.0.weight'] = update_conv_weight_1

#svd_compute_time = end_svd_time - start_svd_time

#print(f"incremental svd compute time:{svd_compute_time} s")

results_df = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Train Accuracy(%)', 'Training Time(s)'])

results_df_1 = pd.DataFrame(columns=['Epoch', 'Test Loss', 'Test Accuracy(%)', 'Test Time(s)'])

NUM_EPOCHS = 101

start_time = time.time()

for epoch in tqdm(range(NUM_EPOCHS)):

	for batch_idx, (features, targets) in enumerate(train_loader):
		### PREPARE MINIBATCH
		features = features.to(DEVICE)
		targets = targets.to(DEVICE)
		### FORWARD AND BACK PROP
		logits, probas = model(features)

		cost = F.cross_entropy(logits, targets)
		optimizer.zero_grad()
		cost.backward()
		### UPDATE MODEL PARAMETERS
		optimizer.step()

		### LOGGING
		if batch_idx % 200 == 0:
			acc = (logits.argmax(1) == targets).float().mean().item()
			print("Train Epochs[{}/{}]---batch[{}/{}]---acc {:.4}---loss {:.4}".format(
				epoch + 1, NUM_EPOCHS, batch_idx, len(train_loader), acc, cost.item()))
		
	elapsed_1 = (time.time() - start_time)
	print(f'Training Time (s)=: {elapsed_1} s')
	# 将训练损失值和测试准确度结果添加到DataFrame中
	results_df = results_df.append({'Epoch': epoch + 1,
		                                'Train Loss': cost.item(),
		                                'Train Accuracy(%)': 100 * acc,
		                                'Training Time(s)': elapsed_1}, ignore_index=True)
		
	# no need to build the computation graph for backprop when computing accuracy
	model.eval()

	valid_acc, valid_loss = compute_accuracy_and_loss(model, valid_loader, device=DEVICE)
	print('Test Epochs[{}/{}]--Loss on Test {:.4}--Accuracy on Test {:.4}'.
	      format(epoch + 1, NUM_EPOCHS, valid_loss, valid_acc))

	elapsed_2 = (time.time() - start_time - elapsed_1)
	print(f'Predit Time (s)=: {elapsed_2} s')
	
	# 将训练损失值和测试准确度结果添加到DataFrame中
	results_df_1 = results_df_1.append({'Epoch': epoch + 1,
	                                    'Test Loss': valid_loss,
	                                    'Test Accuracy(%)': valid_acc.cpu().numpy(),
	                                    'Test Time(s)': elapsed_2}, ignore_index=True)
	
	model.load_state_dict(w_global)

	# 将结果保存到 Excel文件
	with pd.ExcelWriter('100_resnet_cifar_ISVD_fedavg_train_results.xlsx') as writer_1:
		results_df.to_excel(writer_1, index=False, sheet_name='Results')

	with pd.ExcelWriter('100_resnet_cifar_ISVD_fedavg_test_results.xlsx') as writer_2:
		results_df_1.to_excel(writer_2, index=False, sheet_name='Results')

elapsed = (time.time() - start_time) / 60
print(f'Total Training Time: {elapsed:.2f} min')
