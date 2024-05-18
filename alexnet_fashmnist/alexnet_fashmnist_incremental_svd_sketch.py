import csv
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import openpyxl
from tqdm import tqdm


from alexnet_fashmnist_Net import AlexNet
from alexnet_fashmnist_Fed import k_svd_3, FFD_2, rand_hashing, countsketch




train_dataset = datasets.FashionMNIST(root='data',train=True,download=True,transform=ToTensor())

test_dataset = datasets.FashionMNIST(root='data',train=False,download=True,transform=ToTensor())

batch_size = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size)

test_loader = DataLoader(test_dataset, batch_size=batch_size)

device_1 = "cuda" if torch.cuda.is_available() else "cpu"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using {} device".format(device_1))




model = AlexNet(num_classes=10, init_weights=True).to(device)
#print(model)

w_global = model.state_dict()
#print(summary(model))
#print(w_global)



# 获取第二层权重矩阵
weight_fc2 = w_global['features.0.weight'].detach().cpu().numpy()

# 示例数据
x_new = torch.ones(32, 1, 3, 3).detach().cpu().numpy()  # 新数据，与第二层的输出大小相匹配


# 对权重参数进行增量 SVD
def incremental_svd(param, x_new):
	U, Sigma, Vt = np.linalg.svd(param, full_matrices=False)
	Vt = Vt.T
	# print(param.shape)#10,3,5,5
	p = np.dot(x_new.reshape(32, 9), Vt.reshape(9, 32))
	#print(f"p shape: {p.shape}")  # 64*64
	
	param_tran = param.reshape(-1, param.shape[0])  # 49*64
	#print(f"param tran shape:{param_tran.shape}")
	
	new_param = np.hstack((param_tran.reshape(32, 9), p))
	#print(f"new param shape:{new_param.shape}")
	
	return new_param


# 对第二层权重矩阵进行增量 SVD
updated_weight_fc2 = incremental_svd(weight_fc2, x_new)

updated_weight = updated_weight_fc2[:, :9]

# 更新模型的第二层权重矩阵

list1 = []
raw_avg = np.mean(updated_weight, axis=0)

for index in range(updated_weight.shape[0]):
	g = updated_weight[index, :] - raw_avg
	list1.append(g)

array_list = np.array(list1).reshape(32, 9)#
result = FFD_2(array_list)

#print(result.shape)#

x = k_svd_3(result, k=9).reshape(9,9)#9*1*3*3

hash_idx, rand_sgn = rand_hashing(9, 2)

countsketch_x = countsketch(x, hash_idx, rand_sgn)#9*4

tensor_x = torch.cat((countsketch_x, countsketch_x), 1)#9*8

tensor_x_1 = torch.cat((tensor_x, tensor_x), 1)#9*16

b = torch.zeros((9,16))
tensor_x_2 = torch.cat((tensor_x_1, b), 1).reshape(32,1,3,3)#9*32


w_global['features.0.weight'] = tensor_x_2


#svd_compute_time = end_svd_time - start_svd_time

#print(f"incremental svd compute time:{svd_compute_time} s")



criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

results_df = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Train Accuracy(%)', 'Training Time(s)'])

results_df_1 = pd.DataFrame(columns=['Epoch', 'Test Loss', 'Test Accuracy(%)', 'Test Time(s)'])

num_epochs = 101

print('Training Started')

start_time = time.time()

for epoch in tqdm(range(num_epochs)):

	#print(f"Epoch {epoch + 1}\n-----------------")

	running_acc = 0.0

	for batch, (features, labels) in enumerate(train_loader):

		features, labels = features.to(device), labels.to(device)

		# forward propagation
		outputs = model(features)

		# loss
		loss = criterion(outputs, labels)

		_, pred = torch.max(outputs.data, 1)

		accuracy = (pred == labels).float().sum().item()

		running_acc += accuracy

		train_acc = running_acc / (len(train_loader.dataset))


		# initial gradient
		optimizer.zero_grad()

		# backward propagation
		loss.backward()

		# optimise
		optimizer.step()

	print('Epoch [{}/{}], Train Loss: {:.6f}, Train Acc: {:.6f}'.format(epoch + 1, num_epochs,
	                                                                    round(float(loss.item()), 4), train_acc))

	elapsed_1 = (time.time() - start_time)
	train_time = elapsed_1
	print(f'Training Time (s)=: {elapsed_1} s')

	
	# 将训练损失值和测试准确度结果添加到DataFrame中
	results_df = results_df.append({'Epoch': epoch + 1,
	                                'Train Loss': round(float(loss.item()), 4),
	                                'Train Accuracy(%)': 100 * train_acc,
	                                'Training Time(s)': elapsed_1}, ignore_index=True)
	
	#evaluate the model
	model.eval()

	with torch.no_grad():

		correct = 0.0

		for images, labels in test_loader:

			images, labels = images.to(device), labels.to(device)

			outputs = model(images)

			_, pred = torch.max(outputs.data, 1)

			loss_1 = criterion(outputs, labels)

			correct += (pred == labels).float().sum().item()

		test_acc = correct / len(test_dataset)

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
	with pd.ExcelWriter('100_alexnet_fmnist_ISVD_sketch_fedavg_train_results.xlsx') as writer_1:
		results_df.to_excel(writer_1, index=False, sheet_name='Results')
	
	with pd.ExcelWriter('100_alexnet_fmnist_ISVD_sketch_fedavg_test_results.xlsx') as writer_2:
		results_df_1.to_excel(writer_2, index=False, sheet_name='Results')
	
	model.load_state_dict(w_global)




print('Training Finished')


elapsed = (time.time() - start_time) / 60
print(f'Total Training Time: {elapsed:.2f} min')

# Save Models
#torch.save(model.state_dict(), "alexnet_fashmnist_svd.pt")
#print("Saved PyTorch Model State to AlexNet-FashionMnist.pth")































