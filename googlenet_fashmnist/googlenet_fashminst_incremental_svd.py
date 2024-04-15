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
from google_fashmnist_Fed import FFD, k_svd


# import wandb
#
# wandb.init(project="my-tran-project", entity="zhangpan")
# wandb.config = {
#   "learning_rate": 0.001,
#   "epochs": 100,
#   "batch_size": 128
# }




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
print(w_global.keys())
print(len(w_global.keys()))

count = 0

for weight in w_global.keys():
	name = w_global[f'{weight}']
	print(name.shape)
	num_params = name.numel()
	count += num_params
	
print(count)




start_svd_time = time.time()

#conv_weight = w_global['net.0.weight']#64,1,7,7


# 获取第二层权重矩阵
weight_fc2 = w_global['net.0.weight'].detach().cpu().numpy()

# 示例数据
x_new = torch.randn(64, 1, 7, 7)  # 新数据，与第二层的输出大小相匹配


# 对权重参数进行增量 SVD
def incremental_svd(param, x_new):
	U, Sigma, Vt = np.linalg.svd(param, full_matrices=False)
	Vt = Vt.T
	# print(param.shape)#10,3,5,5
	p = np.dot(x_new.reshape(64, 49), Vt.reshape(49, 64))
	print(f"p shape: {p.shape}")  # 64*64
	
	param_tran = param.reshape(-1, param.shape[0])  # 49*64
	print(f"param tran shape:{param_tran.shape}")
	
	new_param = np.hstack((param_tran.reshape(64, 49), p))
	print(f"new param shape:{new_param.shape}")
	
	return new_param


# 对第二层权重矩阵进行增量 SVD
updated_weight_fc2 = incremental_svd(weight_fc2, x_new.numpy())
end_svd_time = time.time()

# 更新模型的第二层权重矩阵
w_global['net.0.weight'].data = torch.from_numpy(updated_weight_fc2)

svd_compute_time = end_svd_time - start_svd_time

print(f"incremental svd compute time:{svd_compute_time} s")


# # train
# lr = 0.001
#
# num_epochs = 101
#
# optimizer = optim.Adam(model.parameters(), lr=lr)
#
# #criterion = nn.CrossEntropyLoss()
# criterion = nn.CrossEntropyLoss()
#
# print('start train on', device)
#
# f1 = open('/home/user/tmp/GoogleNet_CIFAR10/googlenet_fashmnist/svd/google_fashmnist_svd_train_loss_acc.csv', 'w')
# csv_writer1 = csv.writer(f1)
#
# line1_1 = ['epoch', 'train loss', 'train acc', 'train time']
# csv_writer1.writerow(line1_1)
#
#
# f2 = open('/home/user/tmp/GoogleNet_CIFAR10/googlenet_fashmnist/svd/google_fashmnist_svd_test_loss_acc.csv', 'w')
# csv_writer2 = csv.writer(f2)
#
# line2_1 = ['epoch', 'test loss', 'test acc', 'test time']
# csv_writer2.writerow(line2_1)
#
#
#
#
#
# start_time = time.time()
#
# for epoch in range(num_epochs):
#
# 	#train_acc = torch.tensor([.0], dtype=torch.float, device=device)
#
# 	running_acc = 0.0
#
# 	for images, labels in tqdm(train_loader):
#
# 		images, labels = images.to(device), labels.to(device)
#
# 		#logits, probas = model(images)
# 		outputs = model(images)
#
# 		loss = criterion(outputs, labels)
#
# 		_, pred = torch.max(outputs.data, 1)
#
# 		#num_correct = (pred == labels).sum()
#
# 		accuracy = (pred == labels).sum().item()
#
# 		running_acc += accuracy
#
# 		train_acc = 100 * running_acc / (len(train_loader.dataset))
#
# 		optimizer.zero_grad()
#
# 		loss.backward()
#
# 		optimizer.step()
#
#
# 	print('Epoch [{}/{}], Train Loss: {:.6f}, Train Acc: {:.6f}'.format(epoch + 1, num_epochs, round(float(loss.item()), 4), train_acc))
#
# 	elapsed_1 = (time.time() - start_time)
# 	train_time = elapsed_1 / 60
# 	print(f'Training Time (minutes)=: {elapsed_1 / 60:.4f} min')
#
# 	# line1 = [epoch + 1, round(float(loss.item()), 4), train_acc, round(float(train_time), 4)]
# 	#
# 	# csv_writer1.writerow(line1)
#
# 	#eval_loss = 0.0
#
# 	#Test the model
# 	model.eval()
#
# 	with torch.no_grad():
#
# 		correct = 0.0
#
# 		for images, labels in test_loader:
#
# 			images = images.to(device)
#
# 			labels = labels.to(device)
#
# 			outputs = model(images)
#
# 			_, pred = torch.max(outputs.data, 1)
#
# 			loss_1 = criterion(outputs, labels)
# 			#eval_loss += loss_1.item() * labels.size(0)
#
# 			correct += (pred == labels).float().sum().item()
#
# 		#test_loss = eval_loss / len(test_dataset)
# 		test_acc = correct / len(test_dataset)
#
# 	elapsed_2 = (time.time() - start_time - elapsed_1) / 60
#
# 	print(f'Predit Time (minutes)=: {elapsed_2:.4f} min')
#
# 	print('Epoch [{}/{}],Test Loss: {:.4f}, Test Acc: {:.4f}'.format(epoch + 1, num_epochs, round(float(loss_1.item()), 4), test_acc))
# 	# print('Accuracy of the model on the test images: {}'.format(correct / total))
#
# 	# line2 = [epoch + 1, round(float(loss_1.item()), 4), test_acc, round(float(elapsed_2), 4)]
# 	#
# 	# csv_writer2.writerow(line2)
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
# # save the model
# elapsed = (time.time() - start_time) / 60
# print(f'Total Training Time: {elapsed:.2f} min')
#
# # save model
# torch.save(model.state_dict(), './goolenet_fashmnist_svd.pt')


























