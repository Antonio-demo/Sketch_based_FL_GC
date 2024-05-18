import csv
import time
import numpy as np
import torch
import torch.nn as nn
from  torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from alexnet_cifar_Net import AlexNet
from alexnet_cifar_Fed import k_svd_3, FFD_2, rand_hashing, countsketch
import pandas as pd
import openpyxl





# cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define hyper parametes
batchSize = 128





# Load train dataset and test dataset
train_dataset = datasets.CIFAR10(root='data/CIFAR/', train=True, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)

test_dataset = datasets.CIFAR10(root='data/CIFAR/', train=False, download=True, transform=transforms.ToTensor())

test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)



# train model

model = AlexNet().to(device)


w_global = model.state_dict()

# print(w_global.keys())
# print(len(w_global.keys()))


# 获取第二层权重矩阵
weight_fc2 = w_global['layer1.0.weight'].detach().cpu().numpy()

# 示例数据
x_new = torch.ones(32, 3, 3, 3).detach().cpu().numpy()  # 新数据，与第二层的输出大小相匹配


# 对权重参数进行增量 SVD
def incremental_svd(param, x_new):
	U, Sigma, Vt = np.linalg.svd(param, full_matrices=False)
	Vt = Vt.T
	# print(param.shape)#10,3,5,5
	p = np.dot(x_new.reshape(32, 27), Vt.reshape(27, 32))
	#print(f"p shape: {p.shape}")  #
	
	param_tran = param.reshape(-1, param.shape[0])  # 27*32
	#print(f"param tran shape:{param_tran.shape}")
	
	new_param = np.hstack((param_tran.reshape(32, 27), p))
	#print(f"new param shape:{new_param.shape}")
	
	return new_param


# 对第二层权重矩阵进行增量 SVD
updated_weight_fc2 = incremental_svd(weight_fc2, x_new)

#end_svd_time = time.time()


updated_weight = updated_weight_fc2[:, :27]

list1 = []
raw_avg = np.mean(updated_weight, axis=0)

for index in range(updated_weight.shape[0]):
	g = updated_weight[index, :] - raw_avg
	list1.append(g)

array_list = np.array(list1).reshape(32, 27)#
result = FFD_2(array_list)
#print(result.shape)#
x = k_svd_3(result, k=16).reshape(16,27)#16*27

hash_idx, rand_sgn = rand_hashing(27,2)
countsketch_x = countsketch(x, hash_idx, rand_sgn)#16*13
#print(countsketch_x.shape)

tensor_x = torch.cat((countsketch_x, countsketch_x),0)#32*13
#print(tensor_x.shape)

b = torch.zeros((32, 14))
tensor_x_1 = torch.cat((tensor_x, b),1).reshape(32,3,3,3)#32*27

w_global['layer1.0.weight'] = tensor_x_1




print('Training Started')

# Define loss function and Optimizer
criterion = nn.CrossEntropyLoss()  # Cross Entropy function

optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer


results_df = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Train Accuracy(%)', 'Training Time(s)'])

results_df_1 = pd.DataFrame(columns=['Epoch', 'Test Loss', 'Test Accuracy(%)', 'Test Time(s)'])

num_epochs = 101

start_time = time.time()

for epoch in tqdm(range(num_epochs)):  # loop over the dataset multiple times
	torch.cuda.empty_cache()
	running_acc = 0.0

	for i, (features, labels) in enumerate(train_loader, 1):

		features, labels = features.to(device), labels.to(device)  # employ features and labels to the GPU

		outputs = model(features)

		loss = criterion(outputs, labels)

		_, pred = torch.max(outputs.data, 1)

		accuracy = (pred == labels).float().sum().item()

		running_acc += accuracy

		train_acc = running_acc / (len(train_loader.dataset))

		# set gradients as 0 value
		optimizer.zero_grad()

		loss.backward()  # back propagation

		optimizer.step()  # optimizer

	print('Epoch [{}/{}], Train Loss: {:.6f}, Train Acc: {:.6f}'.format(epoch + 1, num_epochs, round(float(loss.item()), 4),
	                                                                    train_acc))
	elapsed_1 = (time.time() - start_time)

	train_time = elapsed_1

	print(f'Training Time (s)=: {elapsed_1} s')

	
	# 将训练损失值和测试准确度结果添加到DataFrame中
	results_df = results_df.append({'Epoch': epoch + 1,
	                                'Train Loss': round(float(loss.item()), 4),
	                                'Train Accuracy(%)': 100 * train_acc,
	                                'Training Time(s)': elapsed_1}, ignore_index=True)
	
	with torch.no_grad():

		correct = 0.0

		for (features, labels) in test_loader:

			features, labels = features.to(device), labels.to(device)  # employ features and labels to the GPU

			outputs = model(features)

			_, predicted = torch.max(outputs.data, 1)  # return max element of each row and index

			loss_1 = criterion(outputs, labels)


			correct += (predicted == labels).float().sum().item()

			test_acc = correct / len(test_loader.dataset)

	print('Epoch [{}/{}],Test Loss: {:.4f}, Test Acc: {:.4f}'.format(epoch + 1, num_epochs,
		                                                                 round(float(loss_1.item()), 4), test_acc))

	elapsed_2 = (time.time() - start_time - elapsed_1)
	print(f'Predit Time (s)=: {elapsed_2} s')
	
	# 将训练损失值和测试准确度结果添加到DataFrame中
	results_df_1 = results_df_1.append({'Epoch': epoch + 1,
	                                    'Test Loss': round(float(loss_1.item()), 4),
	                                    'Test Accuracy(%)': 100 * test_acc,
	                                    'Test Time(s)': elapsed_2}, ignore_index=True)
	
	# 将结果保存到 Excel文件
	with pd.ExcelWriter('100_alexnet_cifar_ISVD_sketch_fedavg_train_results.xlsx') as writer_1:
		results_df.to_excel(writer_1, index=False, sheet_name='Results')
	
	with pd.ExcelWriter('100_alexnet_cifar_ISVD_sketch_fedavg_test_results.xlsx') as writer_2:
		results_df_1.to_excel(writer_2, index=False, sheet_name='Results')
	
	model.load_state_dict(w_global)
	



elapsed = (time.time() - start_time) / 60
print(f'Total Training Time: {elapsed:.2f} min')

# save model
#torch.save(model.state_dict(), './alexnet_cifar_svd.pt')

print('Training Finished')






























