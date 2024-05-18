import time
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import pandas as pd
import openpyxl
from Fed import k_svd, FFD, rand_hashing, countsketch
from tqdm import tqdm




# train process
print("\nLOAD DATA\n")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
train_dataset = datasets.MNIST(root='data',
							   train=True,
							   transform=transforms.ToTensor(),
							   download=True)

test_dataset = datasets.MNIST(root='data',
							  train=False,
							  transform=transforms.ToTensor())

##################################################### Using a GPU #####################################################
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

print("\nUSING", device)
if cuda:
	num_dev = torch.cuda.current_device()
	print(torch.cuda.get_device_name(num_dev), "\n")

train_loader = DataLoader(dataset=train_dataset,
						  batch_size=BATCH_SIZE,
						  shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
						 batch_size=BATCH_SIZE,
						 shuffle=False)


def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class ResNet(nn.Module):

	def __init__(self, block, layers, num_classes, grayscale):
		self.inplanes = 64
		if grayscale:
			in_dim = 1
		else:
			in_dim = 3
		super(ResNet, self).__init__()
		self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AvgPool2d(7, stride=1)
		self.fc = nn.Linear(512 * block.expansion, num_classes)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:  # 看上面的信息是否需要卷积修改，从而满足相加条件
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		# because MNIST is already 1x1 here:
		# disable avg pooling
		# x = self.avgpool(x)

		x = x.view(x.size(0), -1)
		logits = self.fc(x)
		
		probas = F.softmax(logits, dim=1)
		return logits, probas

def resnet10(num_classes):
	"""Constructs a ResNet-18 model."""
	model = ResNet(block=BasicBlock,
				   layers=[1,1,1,1],
				   num_classes=num_classes,
				   grayscale=True)
	return model

def resnet18(num_classes):
	"""Constructs a ResNet-18 model."""
	model = ResNet(block=BasicBlock,
				   layers=[2, 2, 2, 2],
				   num_classes=num_classes,
				   grayscale=True)
	return model

def resnet34(num_classes):
	"""Constructs a ResNet-18 model."""
	model = ResNet(block=BasicBlock,
				   layers=[3,4,6,3],
				   num_classes=num_classes,
				   grayscale=True)
	return model

def resnet50(num_classes):
	"""Constructs a ResNet-18 model."""
	model = ResNet(block=BasicBlock,
				   layers=[3,4,6,3],
				   num_classes=num_classes,
				   grayscale=True)
	return model

def resnet101(num_classes):
	"""Constructs a ResNet-18 model."""
	model = ResNet(block=BasicBlock,
				   layers=[3,4,23,3],
				   num_classes=num_classes,
				   grayscale=True)
	return model

def resnet152(num_classes):
	"""Constructs a ResNet-18 model."""
	model = ResNet(block=BasicBlock,
				   layers=[3,8,36,3],
				   num_classes=num_classes,
				   grayscale=True)
	return model

net = resnet18(10)


model = resnet18(num_classes=10)
model = model.to(DEVICE)

# 原先这里选用SGD训练，但是效果很差，换成Adam优化就好了
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
valid_loader = test_loader


def compute_accuracy_and_loss(model, data_loader, device):
	correct_pred, num_examples = 0, 0
	cross_entropy = 0.
	for i, (features, targets) in enumerate(data_loader):
		features = features.to(device)
		targets = targets.to(device)
		logits, probas = model(features)
		cross_entropy += F.cross_entropy(logits, targets, reduction='sum').item()
		_, predicted_labels = torch.max(probas, 1)
		num_examples += targets.size(0)
		correct_pred += (predicted_labels == targets).sum().item()
		#print(correct_pred)
	return correct_pred / num_examples * 100, cross_entropy / num_examples



train_acc_list, valid_acc_list = [], []
train_loss_list, valid_loss_list = [], []


model.train()
#copy weights
w_global = model.state_dict()
#print(w_glob)

# print(w_global.keys())
# print(len(w_global.keys()))
#
# count = 0
#
# for weight in w_global.keys():
# 	name = w_global[f'{weight}']
# 	print(name.shape)
# 	num_params = name.numel()
# 	count += num_params
#
# print(count)



#layer_1_conv1_weight = w_glob['layer1.0.conv1.weight']
#print(layer_1_conv1_weight.shape)#64*64*3*3
# layer_1_conv2_weight = w_glob['layer1.0.conv2.weight']
# print(layer_1_conv2_weight.shape)#
# layer_11_conv1_weight = w_glob['layer1.1.conv1.weight']
# print(layer_11_conv1_weight.shape)
# layer_11_conv2_weight = w_glob['layer1.1.conv2.weight']
# print(layer_11_conv2_weight.shape)
# layer_2_conv1_weight = w_glob['layer2.0.conv1.weight']
# print(layer_2_conv1_weight.shape)
# layer_2_conv2_weight = w_glob['layer2.0.conv2.weight']
# print(layer_2_conv2_weight.shape)
#
# layer_21_conv1_weight = w_glob['layer2.1.conv1.weight']
# print(layer_21_conv1_weight.shape)
# layer_21_conv2_weight = w_glob['layer2.1.conv2.weight']
# print(layer_21_conv2_weight.shape)
# layer_3_conv1_weight = w_glob['layer3.0.conv1.weight']
# print(layer_3_conv1_weight.shape)
# layer_3_conv2_weight = w_glob['layer3.0.conv2.weight']
# print(layer_3_conv2_weight.shape)
# layer_31_conv1_weight = w_glob['layer3.1.conv1.weight']
# print(layer_31_conv1_weight.shape)
#
# layer_31_conv2_weight = w_glob['layer3.1.conv2.weight']
# print(layer_31_conv2_weight.shape)
#
# layer_4_conv1_weight = w_glob['layer4.0.conv1.weight']
# print(layer_4_conv1_weight.shape)
#
# layer_4_conv2_weight = w_glob['layer4.0.conv2.weight']
# print(layer_4_conv2_weight.shape)
# layer_41_conv1_weight = w_glob['layer4.1.conv1.weight']
# print(layer_41_conv1_weight.shape)
# layer_41_conv2_weight = w_glob['layer4.1.conv2.weight']
# print(layer_41_conv2_weight.shape)
#
# fc_weight = w_glob['fc.weight']
# print(fc_weight.shape)



#conv1_weight = w_global['conv1.weight']

#print(conv1_weight.shape)#64*1*7*7

# 获取第二层权重矩阵
weight_fc2 = w_global['conv1.weight'].detach().cpu().numpy()

# 示例数据
x_new = torch.ones(64, 1, 7, 7).detach().cpu().numpy()   # 新数据，与第二层的输出大小相匹配

# 对权重参数进行增量 SVD
def incremental_svd(param, x_new):
	U, Sigma, Vt = np.linalg.svd(param, full_matrices=False)
	Vt = Vt.T
	# print(param.shape)#10,3,5,5
	p = np.dot(x_new.reshape(64, 49), Vt.reshape(49,64))
	#print(f"p shape: {p.shape}")#64*64
	
	param_tran = param.reshape(-1, param.shape[0])  # 49*64
	#print(f"param tran shape:{param_tran.shape}")
	
	new_param = np.hstack((param_tran.reshape(64,49), p))
	#print(f"new param shape:{new_param.shape}")
	
	return new_param

#start_svd_time = time.time()
# 对第二层权重矩阵进行增量 SVD
updated_weight_fc2 = incremental_svd(weight_fc2, x_new)#64*113
#print(updated_weight_fc2.shape)
updated_weight = updated_weight_fc2[:, :49]

list1 = []
raw_avg = np.mean(updated_weight, axis=0)
for index in range(updated_weight.shape[0]):
	g = updated_weight[index, :] - raw_avg
	list1.append(g)
array_list = np.array(list1).reshape(64, 49)
result = FFD(array_list)
# print(result.shape)#64*49
x = k_svd(result, k=16).reshape(16, 49)

hash_idx, rand_sgn = rand_hashing(49, 2)
countsketch_x = countsketch(x, hash_idx, rand_sgn)
#print(countsketch_x.shape)#16*24

b_1 = torch.zeros((16, 25))
tensor_x_1 = torch.cat((countsketch_x, b_1), dim=1)
#print(tensor_x_1.shape)#16*49
b_2 = torch.zeros((48, 49))
tensor_x_2 = torch.cat((tensor_x_1, b_2), dim=0).reshape(64,1,7,7)


w_global['conv1.weight'] = tensor_x_2


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
		if batch_idx % 200==0:
			acc = (logits.argmax(1)==targets).float().mean().item()
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
	#train_acc, train_loss = compute_accuracy_and_loss(model, train_loader, device=DEVICE)
	valid_acc, valid_loss = compute_accuracy_and_loss(model, valid_loader, device=DEVICE)
	
	print("Test Epochs[{}/{}]--Loss on Test {:.4}--Accuracy on Test {:.4}".
		      format(epoch + 1, NUM_EPOCHS,valid_loss, valid_acc / 100))
	
	elapsed_2 = (time.time() - start_time - elapsed_1)
	print(f'Predit Time (s)=: {elapsed_2} s')
	
	# 将训练损失值和测试准确度结果添加到DataFrame中
	results_df_1 = results_df_1.append({'Epoch': epoch + 1,
	                                    'Test Loss': valid_loss,
	                                    'Test Accuracy(%)': valid_acc,
	                                    'Test Time(s)': elapsed_2}, ignore_index=True)
	
	# 将结果保存到 Excel文件
	with pd.ExcelWriter('100_resnet_mnist_ISVD_sketch_fedavg_train_results.xlsx') as writer_1:
		results_df.to_excel(writer_1, index=False, sheet_name='Results')
	
	with pd.ExcelWriter('100_resnet_mnist_ISVD_sketch_fedavg_test_results.xlsx') as writer_2:
		results_df_1.to_excel(writer_2, index=False, sheet_name='Results')
	

	model.load_state_dict(w_global)


elapsed = (time.time() - start_time) / 60
print(f'Total Training Time: {elapsed:.2f} min')



# # plotting process
# #训练损失和测试损失图
# plt.plot(range(1, NUM_EPOCHS + 1), train_loss_list, label='Training loss')
# plt.plot(range(1, NUM_EPOCHS + 1), valid_loss_list, label='Validation loss')
# plt.legend(loc='upper right')
# plt.ylabel('Cross entropy')
# plt.xlabel('Epoch')
# plt.show()
#
# #训练精度和测试精度
# plt.plot(range(1, NUM_EPOCHS + 1), train_acc_list, label='Training accuracy')
# plt.plot(range(1, NUM_EPOCHS + 1), valid_acc_list, label='Validation accuracy')
# plt.legend(loc='upper left')
# plt.ylabel('Cross entropy')
# plt.xlabel('Epoch')
# plt.show()
#
# # test process
# model.eval()
# with torch.set_grad_enabled(False):  # save memory during inference
# 	test_acc, test_loss = compute_accuracy_and_loss(model, test_loader, DEVICE)
# 	print(f'Test accuracy: {test_acc:.2f}%')
#
#
#
# for features, targets in train_loader:
# 	break
# # 预测环节
# _, predictions = model.forward(features[:8].to(DEVICE))
# predictions = torch.argmax(predictions, dim=1)
# print(predictions)
#
# features = features[:7]
# fig = plt.figure()
# # print(features[i].size())
# for i in range(6):
# 	plt.subplot(2, 3, i + 1)
# 	plt.tight_layout()
# 	tmp = features[i]
# 	plt.imshow(np.transpose(tmp, (1, 2, 0)))
# 	plt.title("Actual value: {}".format(targets[i]) + '\n' + "Prediction value: {}".format(predictions[i]), size=10)
#
# #     plt.title("Prediction value: {}".format(tname[targets[i]]))
# plt.show()
# """












