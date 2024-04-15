import csv
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
# import AlexNet


from alexnet_fashmnist_Net import AlexNet
from alexnet_fashmnist_Fed import FFD, k_svd


# import wandb
# wandb.init(project="my-tran-project", entity="zhangpan")
# wandb.config = {
#   "learning_rate": 0.001,
#   "epochs": 100,
#   "batch_size": 128
# }




train_dataset = datasets.FashionMNIST(root='data',train=True,download=True,transform=ToTensor())

test_dataset = datasets.FashionMNIST(root='data',train=False,download=True,transform=ToTensor())

batch_size = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size)

test_loader = DataLoader(test_dataset, batch_size=batch_size)

device_1 = "cuda" if torch.cuda.is_available() else "cpu"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using {} device".format(device_1))


# f1 = open('/home/user/tmp/GoogleNet_CIFAR10/alexnet_fashmnist/svd/alexnet_fashmnist_svd_train_loss_acc.csv', 'w')
# csv_writer1 = csv.writer(f1)
#
# line1_1 = ['epoch', 'train loss', 'train acc', 'train time']
# csv_writer1.writerow(line1_1)
#
#
# f2 = open('/home/user/tmp/GoogleNet_CIFAR10/alexnet_fashmnist/svd/alexnet_fashmnist_svd_test_loss_acc.csv', 'w')
# csv_writer2 = csv.writer(f2)
#
# line2_1 = ['epoch', 'test loss', 'test acc', 'test time']
# csv_writer2.writerow(line2_1)



model = AlexNet(num_classes=10, init_weights=True).to(device)
#print(model)

w_global = model.state_dict()
#print(summary(model))
#print(w_global)

print(w_global.keys())
print(len(w_global.keys()))

count = 0

for weight in w_global.keys():
	name = w_global[f'{weight}']
	print(name.shape)
	num_params = name.numel()
	count += num_params

print(count)



#conv_weight = w_global['features.0.weight']#32*1*3*3
#print(conv_weight.shape)

#array_conv_weight = conv_weight.cpu().numpy().reshape(16,18)

start_svd_time = time.time()

# 获取第二层权重矩阵
weight_fc2 = w_global['features.0.weight'].detach().cpu().numpy()

# 示例数据
x_new = torch.randn(32, 1, 3, 3)  # 新数据，与第二层的输出大小相匹配


# 对权重参数进行增量 SVD
def incremental_svd(param, x_new):
	U, Sigma, Vt = np.linalg.svd(param, full_matrices=False)
	Vt = Vt.T
	# print(param.shape)#10,3,5,5
	p = np.dot(x_new.reshape(32, 9), Vt.reshape(9, 32))
	print(f"p shape: {p.shape}")  # 64*64
	
	param_tran = param.reshape(-1, param.shape[0])  # 49*64
	print(f"param tran shape:{param_tran.shape}")
	
	new_param = np.hstack((param_tran.reshape(32, 9), p))
	print(f"new param shape:{new_param.shape}")
	
	return new_param


# 对第二层权重矩阵进行增量 SVD
updated_weight_fc2 = incremental_svd(weight_fc2, x_new.numpy())

end_svd_time = time.time()

# 更新模型的第二层权重矩阵
w_global['features.0.weight'].data = torch.from_numpy(updated_weight_fc2)

svd_compute_time = end_svd_time - start_svd_time

print(f"incremental svd compute time:{svd_compute_time} s")



# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
#
# num_epochs = 101
#
# print('Training Started')
#
# start_time = time.time()
#
# for epoch in range(num_epochs):
#
# 	#print(f"Epoch {epoch + 1}\n-----------------")
#
# 	running_acc = 0.0
#
# 	for batch, (features, labels) in enumerate(train_loader):
#
# 		features, labels = features.to(device), labels.to(device)
#
# 		# forward propagation
# 		outputs = model(features)
#
# 		# loss
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
#
# 		# initial gradient
# 		optimizer.zero_grad()
#
# 		# backward propagation
# 		loss.backward()
#
# 		# optimise
# 		optimizer.step()
#
# 		# if batch % 100 == 0:
# 		# 	loss, current = loss.item(), batch * len(features)
# 		# 	print(f"loss: {loss: >7f} [{current: >5d}/{size: >5d}]")
#
# 	print('Epoch [{}/{}], Train Loss: {:.6f}, Train Acc: {:.6f}'.format(epoch + 1, num_epochs,
# 	                                                                    round(float(loss.item()), 4), train_acc))
#
# 	elapsed_1 = (time.time() - start_time)
# 	train_time = elapsed_1 / 60
# 	print(f'Training Time (minutes)=: {elapsed_1 / 60:.4f} min')
#
# 	line1 = [epoch + 1, round(float(loss.item()), 4), train_acc, round(float(train_time), 4)]
#
# 	csv_writer1.writerow(line1)
#
#
# 	#evaluate the model
# 	model.eval()
#
# 	with torch.no_grad():
#
# 		correct = 0.0
#
# 		for images, labels in test_loader:
#
# 			images, labels = images.to(device), labels.to(device)
#
# 			outputs = model(images)
#
# 			_, pred = torch.max(outputs.data, 1)
#
# 			loss_1 = criterion(outputs, labels)
#
# 			correct += (pred == labels).float().sum().item()
#
# 		test_acc = correct / len(test_dataset)
#
# 	print('Epoch [{}/{}],Test Loss: {:.4f}, Test Acc: {:.4f}'.format(epoch + 1, num_epochs,
# 	                                  round(float(loss_1.item()), 4), test_acc))
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
#
# print('Training Finished')
#
#
# elapsed = (time.time() - start_time) / 60
# print(f'Total Training Time: {elapsed:.2f} min')
#
# # Save Models
# torch.save(model.state_dict(), "alexnet_fashmnist_svd.pt")
# #print("Saved PyTorch Model State to AlexNet-FashionMnist.pth")































