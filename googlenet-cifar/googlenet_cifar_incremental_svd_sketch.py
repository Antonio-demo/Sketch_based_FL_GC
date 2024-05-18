import csv
import time
import numpy as np
import torch
import torchvision
from tqdm import tqdm
import gc
from google_cifar_Fed import k_svd_2, FFD_2, rand_hashing, countsketch
from Net import GoogLeNet
from torch.utils.data import DataLoader
import pandas as pd
import openpyxl




# Device configuration.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Transform configuration and Data Augmentation.
transform_train = torchvision.transforms.Compose([torchvision.transforms.Pad(2),
                                                  torchvision.transforms.RandomHorizontalFlip(),
                                                  torchvision.transforms.RandomCrop(32),
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# Load downloaded dataset.
train_dataset = torchvision.datasets.CIFAR10('data/CIFAR/', download=False, train=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10('data/CIFAR/', download=False, train=False, transform=transform_test)


# Hyper-parameters

batch_size = 100
num_classes = 10
learning_rate = 0.001

# Data Loader.
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



# Make model.
model = GoogLeNet(num_classes, False, True).to(device)
# model = GoogLeNet(num_classes, True, True).to(device) # Auxiliary Classifier


# Loss and optimizer.
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# For updating learning rate.
def update_lr(optimizer, lr):
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


# Train the model

running_loss, running_acc = 0.0, 0.0

print("\nLOAD DATA\n")

model.train()



w_global = model.state_dict()



# 获取第二层权重矩阵
weight_fc2 = w_global['conv1.conv.weight'].detach().cpu().numpy()

# 示例数据
x_new = torch.ones(64, 3, 4, 4).detach().cpu().numpy()  # 新数据，与第二层的输出大小相匹配


# 对权重参数进行增量 SVD
def incremental_svd(param, x_new):
	U, Sigma, Vt = np.linalg.svd(param, full_matrices=False)
	Vt = Vt.T
	# print(param.shape)#10,3,5,5
	p = np.dot(x_new.reshape(64, 48), Vt.reshape(48, 64))
	#print(f"p shape: {p.shape}")  # 64*64
	
	param_tran = param.reshape(-1, param.shape[0])  # 49*64
	#print(f"param tran shape:{param_tran.shape}")
	
	new_param = np.hstack((param_tran.reshape(64, 48), p))
	#print(f"new param shape:{new_param.shape}")
	
	return new_param


# 对第二层权重矩阵进行增量 SVD
updated_weight_fc2 = incremental_svd(weight_fc2, x_new)

#end_svd_time = time.time()

updated_weight = updated_weight_fc2[:, :48]

list1 = []
raw_avg = np.mean(updated_weight, axis=0)

for index in range(updated_weight.shape[0]):
	g = updated_weight[index, :] - raw_avg
	list1.append(g)

array_list = np.array(list1).reshape(64, 48)
result = FFD_2(array_list)
# print(result.shape)#64*27
x = k_svd_2(result, k=10).reshape(10,48)#10*48

hash_idx, rand_sgn = rand_hashing(48,2)

countsketch_x = countsketch(x, hash_idx, rand_sgn)#10*24

#print(countsketch_x.shape)
tensor_x = torch.cat((countsketch_x, countsketch_x), 1)#10*48
#print(tensor_x.shape)
tensor_x_1 = torch.cat((tensor_x, tensor_x), 0)#20*48

tensor_x_2 = torch.cat((tensor_x_1, tensor_x_1),0)#40*48

tensor_x_3 = torch.cat((tensor_x_2, tensor_x_2),0)#80*48

array_tensor_3 = tensor_x_3.numpy()


array_tensor_3 = array_tensor_3[:64,:]
tensor_x_4 = torch.from_numpy(array_tensor_3).reshape(64,3,4,4)#64*48


b = torch.zeros((44, 48))

tensor_x_2 = torch.cat((tensor_x_1, b),0).reshape(64,3,4,4)#64*48
#print(tensor_x_2.shape)

w_global['conv1.conv.weight'] = tensor_x_4

#svd_compute_time = end_svd_time - start_svd_time

#print(f"incremental svd compute time:{svd_compute_time} s")



total_step = len(train_loader)
curr_lr = learning_rate



results_df = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Train Accuracy(%)', 'Training Time(s)'])

results_df_1 = pd.DataFrame(columns=['Epoch', 'Test Loss', 'Test Accuracy(%)', 'Test Time(s)'])

num_epochs = 101 # To decrease the training time of model.



start_time = time.time()

for epoch in tqdm(range(num_epochs)):
	gc.collect()
	torch.cuda.empty_cache()
	total_num = 0

	for i, (images, labels) in enumerate(train_loader, 1):

		images = images.to(device)

		labels = labels.to(device)

		# Forward pass
		outputs = model(images)
		loss = criterion(outputs, labels)

		_, pred = torch.max(outputs.data, 1)  # 预测最大值所在的位置标签

		accuracy = (pred == labels).float().sum().item()

		running_acc += accuracy

		#train_loss = running_loss / (100 * len(train_dataset))

		train_acc = running_acc / (100 * len(train_loader))

		# Backward and optimize
		optimizer.zero_grad()

		loss.backward()

		optimizer.step()

		# if (i + 1) % 100 == 0:
		#     print('Epoch [{}/{}], Step [{}/{}], Loss {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step,
		#                                                             loss.item()))
	print('Epoch [{}/{}], Train Loss: {:.6f}, Train Acc: {:.6f}'.format(epoch + 1, num_epochs, loss.item(), train_acc))

	
	

	# Decay learning rate
	if (epoch + 1) % 20 == 0:
		curr_lr /= 3
		update_lr(optimizer, curr_lr)

	eval_loss = 0.0
	# Test the mdoel.
	model.eval()
	
	elapsed_1 = (time.time() - start_time)
	train_time = elapsed_1
	print(f'Training Time (s)=: {elapsed_1} s')
	
	# 将训练损失值和测试准确度结果添加到DataFrame中
	results_df = results_df.append({'Epoch': epoch + 1,
	                                'Train Loss': round(float(loss.item()), 4),
	                                'Train Accuracy(%)': 100 * train_acc,
	                                'Training Time(s)': elapsed_1}, ignore_index=True)
	
	
	with torch.no_grad():
		correct = 0
		total = 0
		for images, labels in test_loader:
			images = images.to(device)

			labels = labels.to(device)

			outputs = model(images)

			_, predicted = torch.max(outputs.data, 1)

			loss_1 = criterion(outputs, labels)
			#eval_loss += loss.item() * labels.size(0)

			correct += (predicted == labels).float().sum().item()

		#test_loss = eval_loss / len(test_dataset)
		test_acc = correct / len(test_dataset)


	elapsed_2 = (time.time() - start_time - elapsed_1)
	print(f'Predit Time (s)=: {elapsed_2} s')

	print('Epoch [{}/{}],Test Loss: {:.4f}, Test Acc: {:.4f}'.format(epoch+1, num_epochs, round(float(loss_1.item()), 4), test_acc))
	#print('Accuracy of the model on the test images: {}'.format(correct / total))
	
	# 将训练损失值和测试准确度结果添加到DataFrame中
	results_df_1 = results_df_1.append({'Epoch': epoch + 1,
	                                    'Test Loss': round(float(loss_1.item()), 4),
	                                    'Test Accuracy(%)': 100 * test_acc,
	                                    'Test Time(s)': elapsed_2}, ignore_index=True)
	
	# 将结果保存到 Excel文件
	with pd.ExcelWriter('100_googlenet_cifar_ISVD_sketch_fedavg_train_results.xlsx') as writer_1:
		results_df.to_excel(writer_1, index=False, sheet_name='Results')
	
	with pd.ExcelWriter('100_googlenet_cifar_ISVD_sketch_fedavg_test_results.xlsx') as writer_2:
		results_df_1.to_excel(writer_2, index=False, sheet_name='Results')
	
	model.load_state_dict(w_global)

elapsed = (time.time() - start_time) / 60
print(f'Total Training Time: {elapsed:.2f} min')
# save model
#torch.save(model.state_dict(), './goolenet_cifar_svd.pt')






























