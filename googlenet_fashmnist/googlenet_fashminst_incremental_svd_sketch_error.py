import csv
import time

import numpy as np
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from googlenet_fashmnist_Net import GoogleNet
from google_fashmnist_Fed import k_svd_2, FFD_2, rand_hashing, countsketch
import pandas as pd
import openpyxl




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 128

# load dataset
transforms = transforms.Compose([transforms.Resize(96),
                                 transforms.ToTensor(),
                                 ])


train_dataset = torchvision.datasets.FashionMNIST('data', True, transforms, download=True)

test_dataset = torchvision.datasets.FashionMNIST('data', False, transforms, download=True)

train_loader = DataLoader(train_dataset, batch_size, True)

test_loader = DataLoader(test_dataset, batch_size, True)



# google net
model = GoogleNet(1,10)

model = model.to(device)

# 打印模型的网络架构
#print(model)


w_global = model.state_dict()
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




#start_svd_time = time.time()

#conv_weight = w_global['net.0.weight']#64,1,7,7


# 获取第二层权重矩阵
weight_fc2 = w_global['net.0.weight'].detach().cpu().numpy()

# 示例数据
x_new = torch.ones(64, 1, 7, 7).detach().cpu().numpy()  # 新数据，与第二层的输出大小相匹配


# 对权重参数进行增量 SVD
def incremental_svd(param, x_new):
	U, Sigma, Vt = np.linalg.svd(param, full_matrices=False)
	Vt = Vt.T
	# print(param.shape)#10,3,5,5
	p = np.dot(x_new.reshape(64, 49), Vt.reshape(49, 64))
	#print(f"p shape: {p.shape}")  # 64*64
	
	param_tran = param.reshape(-1, param.shape[0])  # 49*64
	#print(f"param tran shape:{param_tran.shape}")
	
	new_param = np.hstack((param_tran.reshape(64, 49), p))
	#print(f"new param shape:{new_param.shape}")
	
	return new_param


# 对第二层权重矩阵进行增量 SVD
updated_weight_fc2 = incremental_svd(weight_fc2, x_new)
#end_svd_time = time.time()

updated_weight = updated_weight_fc2[:, :49]


list1 = []

raw_avg = np.mean(updated_weight, axis=0)

for index in range(updated_weight.shape[0]):
	g = updated_weight[index, :] - raw_avg
	list1.append(g)


array_list = np.array(list1).reshape(64,49)

result = FFD_2(array_list)
x = k_svd_2(result, k=32).reshape(32,49)

hash_idx, rand_sgn = rand_hashing(49,2)

countsketch_x = countsketch(x, hash_idx, rand_sgn)#32*24
b_1 = torch.zeros((32, 25))
countsketch_x_2 = torch.cat((countsketch_x, b_1), 1)#32*49


countsketch_x_1 = torch.cat((countsketch_x, countsketch_x), 0)#64*24
b = torch.zeros((64,25))
tensor_x = torch.cat((countsketch_x_1, b), 1).reshape(64, 1, 7, 7)#64*49

alpha = 0.1
error_accumulate = x - countsketch_x_2 #32*49

error_accumulate_1 = torch.cat((error_accumulate, error_accumulate), 0).reshape(64, 1, 7, 7)#64*49

tensor_x_1 = tensor_x + alpha * error_accumulate_1

w_global['net.0.weight'] = tensor_x_1





# train
lr = 0.001

optimizer = optim.Adam(model.parameters(), lr=lr)

#criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss()

print('start train on', device)


results_df = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Train Accuracy(%)', 'Training Time(s)'])

results_df_1 = pd.DataFrame(columns=['Epoch', 'Test Loss', 'Test Accuracy(%)', 'Test Time(s)'])

num_epochs = 101

start_time = time.time()

for epoch in tqdm(range(num_epochs)):
	running_acc = 0.0

	for images, labels in train_loader:

		images, labels = images.to(device), labels.to(device)

		#logits, probas = model(images)
		outputs = model(images)

		loss = criterion(outputs, labels)

		_, pred = torch.max(outputs.data, 1)

		accuracy = (pred == labels).sum().item()

		running_acc += accuracy

		train_acc = running_acc / (len(train_loader.dataset))

		optimizer.zero_grad()

		loss.backward()

		optimizer.step()


	print('Epoch [{}/{}], Train Loss: {:.6f}, Train Acc: {:.6f}'.format(epoch + 1, num_epochs, round(float(loss.item()), 4), train_acc))

	elapsed_1 = (time.time() - start_time)
	train_time = elapsed_1
	print(f'Training Time (s)=: {elapsed_1} s')
	
	# 将训练损失值和测试准确度结果添加到DataFrame中
	results_df = results_df.append({'Epoch': epoch + 1,
	                                'Train Loss': round(float(loss.item()), 4),
	                                'Train Accuracy(%)': 100 * train_acc,
	                                'Training Time(s)': elapsed_1}, ignore_index=True)

	#eval_loss = 0.0

	#Test the model
	model.eval()

	with torch.no_grad():

		correct = 0.0

		for images, labels in test_loader:

			images = images.to(device)

			labels = labels.to(device)

			outputs = model(images)

			_, pred = torch.max(outputs.data, 1)

			loss_1 = criterion(outputs, labels)
			#eval_loss += loss_1.item() * labels.size(0)

			correct += (pred == labels).float().sum().item()

		#test_loss = eval_loss / len(test_dataset)
		test_acc = correct / len(test_dataset)

	elapsed_2 = (time.time() - start_time - elapsed_1)

	print(f'Predit Time (s)=: {elapsed_2} s')

	print('Epoch [{}/{}],Test Loss: {:.4f}, Test Acc: {:.4f}'.format(epoch + 1, num_epochs, round(float(loss_1.item()), 4), test_acc))
	# print('Accuracy of the model on the test images: {}'.format(correct / total))

	# 将训练损失值和测试准确度结果添加到DataFrame中
	results_df_1 = results_df_1.append({'Epoch': epoch + 1,
	                                    'Test Loss': round(float(loss_1.item()), 4),
	                                    'Test Accuracy(%)': 100 * test_acc,
	                                    'Test Time(s)': elapsed_2}, ignore_index=True)
	
	# 将结果保存到 Excel文件
	with pd.ExcelWriter('100_googlenet_fmnist_ISVD_sketch_error_fedavg_train_results.xlsx') as writer_1:
		results_df.to_excel(writer_1, index=False, sheet_name='Results')
	
	with pd.ExcelWriter('100_googlenet_fmnist_ISVD_sketch_error_fedavg_test_results.xlsx') as writer_2:
		results_df_1.to_excel(writer_2, index=False, sheet_name='Results')
	
	model.load_state_dict(w_global)
	




# save the model
elapsed = (time.time() - start_time) / 60
print(f'Total Training Time: {elapsed:.2f} min')

# save model
#torch.save(model.state_dict(), './goolenet_fashmnist_svd.pt')


























