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
from alexnet_cifar_Fed import FFD, k_svd


# import wandb
#
# wandb.init(project="my-tran-project", entity="zhangpan")
# wandb.config = {
#   "learning_rate": 0.001,
#   "epochs": 100,
#   "batch_size": 128
# }




# cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define hyper parametes
batchSize = 128

num_epochs = 101



# Load train dataset and test dataset
train_dataset = datasets.CIFAR10(root='data/CIFAR/', train=True, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)

test_dataset = datasets.CIFAR10(root='data/CIFAR/', train=False, download=True, transform=transforms.ToTensor())

test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)

# f1 = open('/home/user/tmp/GoogleNet_CIFAR10/alexnet_cifar/svd/alex_cifar_svd_train_loss_acc.csv', 'w')
# csv_writer1 = csv.writer(f1)
#
# line1_1 = ['epoch', 'train loss', 'train acc', 'train time']
# csv_writer1.writerow(line1_1)
#
#
# f2 = open('/home/user/tmp/GoogleNet_CIFAR10/alexnet_cifar/svd/alex_cifar_svd_test_loss_acc.csv', 'w')
# csv_writer2 = csv.writer(f2)
#
# line2_1 = ['epoch', 'test loss', 'test acc', 'test time']
# csv_writer2.writerow(line2_1)


# train model

model = AlexNet().to(device)





w_global = model.state_dict()

print(w_global.keys())
print(len(w_global.keys()))

count = 0

for weight in w_global.keys():
	name = w_global[f'{weight}']
	print(name.shape)
	num_params = name.numel()
	count += num_params

print(count)


#layer_weight = w_global['layer1.0.weight']#32*3*3*3

#print(layer_weight.shape)

#array_conv_weight = layer_weight.cpu().numpy().reshape(32,27)

start_svd_time = time.time()

# 获取第二层权重矩阵
weight_fc2 = w_global['layer1.0.weight'].detach().cpu().numpy()

# 示例数据
x_new = torch.randn(32, 3, 3, 3)  # 新数据，与第二层的输出大小相匹配


# 对权重参数进行增量 SVD
def incremental_svd(param, x_new):
	U, Sigma, Vt = np.linalg.svd(param, full_matrices=False)
	Vt = Vt.T
	# print(param.shape)#10,3,5,5
	p = np.dot(x_new.reshape(32, 27), Vt.reshape(27, 32))
	print(f"p shape: {p.shape}")  #
	
	param_tran = param.reshape(-1, param.shape[0])  # 27*32
	print(f"param tran shape:{param_tran.shape}")
	
	new_param = np.hstack((param_tran.reshape(32, 27), p))
	print(f"new param shape:{new_param.shape}")
	
	return new_param


# 对第二层权重矩阵进行增量 SVD
updated_weight_fc2 = incremental_svd(weight_fc2, x_new.numpy())

end_svd_time = time.time()

# 更新模型的第二层权重矩阵
w_global['layer1.0.weight'].data = torch.from_numpy(updated_weight_fc2)

svd_compute_time = end_svd_time - start_svd_time

print(f"incremental svd compute time:{svd_compute_time} s")




# print('Training Started')
#
# # Define loss function and Optimizer
# criterion = nn.CrossEntropyLoss()  # Cross Entropy function
#
# optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
#
#
#
#
#
# start_time = time.time()
#
# for epoch in range(num_epochs):  # loop over the dataset multiple times
# 	torch.cuda.empty_cache()
# 	model.train()
#
# 	running_acc = 0.0
#
# 	for i, (features, labels) in tqdm(enumerate(train_loader, 1)):
#
# 		features, labels = features.to(device), labels.to(device)  # employ features and labels to the GPU
#
# 		outputs = model(features)
#
# 		loss = criterion(outputs, labels)
#
# 		_, pred = torch.max(outputs.data, 1)
#
# 		accuracy = (pred == labels).float().sum().item()
#
# 		running_acc += accuracy
#
# 		train_acc = 100 * running_acc / (len(train_loader.dataset))
#
# 		# set gradients as 0 value
# 		optimizer.zero_grad()
#
# 		loss.backward()  # back propagation
#
# 		optimizer.step()  # optimizer
#
# 	print('Epoch [{}/{}], Train Loss: {:.6f}, Train Acc: {:.6f}'.format(epoch + 1, num_epochs, round(float(loss.item()), 4),
# 	                                                                    train_acc))
# 	elapsed_1 = (time.time() - start_time)
#
# 	train_time = elapsed_1 / 60
#
# 	print(f'Training Time (minutes)=: {elapsed_1 / 60:.4f} min')
#
# 	line1 = [epoch + 1, round(float(loss.item()), 4), train_acc, round(float(train_time), 4)]
#
# 	csv_writer1.writerow(line1)
#
#
# 	with torch.no_grad():
#
# 		correct = 0.0
#
# 		for (features, labels) in test_loader:
#
# 			features, labels = features.to(device), labels.to(device)  # employ features and labels to the GPU
#
# 			outputs = model(features)
#
# 			_, predicted = torch.max(outputs.data, 1)  # return max element of each row and index
#
# 			loss_1 = criterion(outputs, labels)
#
#
# 			correct += (predicted == labels).float().sum().item()
#
# 			test_acc = 100.0 * correct / len(test_loader.dataset)
#
# 	print('Epoch [{}/{}],Test Loss: {:.4f}, Test Acc: {:.4f}'.format(epoch + 1, num_epochs,
# 		                                                                 round(float(loss_1.item()), 4), test_acc))
#
# 	elapsed_2 = (time.time() - start_time - elapsed_1) / 60
# 	print(f'Predit Time (minutes)=: {elapsed_2:.4f} min')
#
# 	line2 = [epoch + 1, round(float(loss_1.item()), 4), test_acc, round(float(elapsed_2), 4)]
#
# 	csv_writer2.writerow(line2)
#
# 	# wandb.log({"Train loss": round(float(loss.item()), 4), 'epoch': epoch + 1})
# 	# wandb.log({"Train acc": train_acc, 'epoch': epoch + 1})
# 	# wandb.log({"Test loss": round(float(loss_1.item()), 4), 'epoch': epoch + 1})
# 	# wandb.log({"Test acc": test_acc, 'epoch': epoch + 1})
# 	# wandb.log({"Train time": elapsed_1 / 60, 'epoch': epoch + 1})
# 	# wandb.log({"Predit time": elapsed_2, 'epoch': epoch + 1})
#
#
# elapsed = (time.time() - start_time) / 60
# print(f'Total Training Time: {elapsed:.2f} min')
#
# # save model
# torch.save(model.state_dict(), './alexnet_cifar_svd.pt')
#
# print('Training Finished')





























