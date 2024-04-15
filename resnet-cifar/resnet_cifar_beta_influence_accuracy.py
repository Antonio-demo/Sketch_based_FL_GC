import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from Fed import FFD_2, k_svd_2, rand_hashing, countsketch

import wandb

wandb.init(project="my-tran-project", entity="zhangpan")
wandb.config = {
	"learning_rate": 0.001,
	"epochs": 100,
	"batch_size": 128
}

"""
"""


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

NUM_EPOCHS = 100

start_time = time.time()

for number in range(1, 11, 1):
	model.train()
	w_glob = model.state_dict()
	# print(w_glob)
	conv1_weight = w_glob['conv1.0.weight']
	# print(conv1_weight.shape)#(64, 3, 3, 3)
	
	array_conv1_weight = conv1_weight.cpu().numpy().reshape(64, 27)
	list1 = []
	raw_avg = np.mean(array_conv1_weight, axis=0)
	for index in range(array_conv1_weight.shape[0]):
		g = array_conv1_weight[index, :] - raw_avg
		list1.append(g)
	array_list = np.array(list1).reshape(64, 27)
	result = FFD_2(array_list)
	# print(result.shape)#(64, 27)
	x = k_svd_2(result, k=20).reshape(20, 27)
	# print(x.shape)#(20,3,3,3)
	
	hash_idx, rand_sgn = rand_hashing(27, 2)
	countsketch_x = countsketch(x, hash_idx, rand_sgn)
	# print(countsketch_x.shape)#(20,13)
	
	b = torch.zeros((20, 14))
	# # print(b.shape)#64*1*4*7
	tensor_x = torch.cat((countsketch_x, b), 1)
	# print(tensor_x.shape)#(64,3,3,3)
	b_2 = torch.zeros((44, 27))
	tensor_x_2 = torch.cat((tensor_x, b_2), 0).reshape(64, 3, 3, 3)
	# print(tensor_x_2.shape)

	beta= number / 10
	error_accumulate = x - tensor_x
	# print(error_accumulate.shape)
	alpha = 1
	countsketch_x = tensor_x + alpha * error_accumulate + beta * alpha * error_accumulate
	w_glob['conv1.0.weight'] = tensor_x_2
	print(f"=================================beta：{beta}=========================================")
	
	for epoch in range(NUM_EPOCHS):
		
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
			if batch_idx != 0 and batch_idx % 300 == 0:
				acc = (logits.argmax(1) == targets).float().mean().item()
				print("Train Epochs[{}/{}]---batch[{}/{}]---acc {:.4}---loss {:.4}".format(
					epoch + 1, NUM_EPOCHS, batch_idx, len(train_loader), acc * 100, cost.item()))

		elapsed_1 = (time.time() - start_time)
		print(f'Training Time (minutes)=: {elapsed_1 / 60:.4f} min')
		# no need to build the computation graph for backprop when computing accuracy
		model.eval()

		valid_acc, valid_loss = compute_accuracy_and_loss(model, valid_loader, device=DEVICE)
		print('Test Epochs[{}/{}]--Loss on Test {:.4}--Accuracy on Test {:.4}'.
		      format(epoch + 1, NUM_EPOCHS, valid_loss, valid_acc ))

		model.load_state_dict(w_glob)
		wandb.log({"Training time (minutes)": elapsed_1 / 60, "epoch": epoch + 1})

	wandb.log({"Train acc": acc * 100, 'beta': beta})
	wandb.log({"Test acc": valid_acc , 'beta': beta})










