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

start_svd_time = time.time()
layer_weight = w_global['layer1.0.weight']#32*3*3*3

#print(layer_weight.shape)


array_conv_weight = layer_weight.cpu().numpy().reshape(32,27)


list1 = []
raw_avg = np.mean(array_conv_weight, axis=0)

for index in range(array_conv_weight.shape[0]):
	g = array_conv_weight[index, :] - raw_avg
	list1.append(g)

array_list = np.array(list1).reshape(32, 27)#
result = FFD(array_list)
#print(result.shape)#
x = k_svd(result, k=16)#16*27

# # pad 0 value
# print(x.shape)#10*3*4*4
b = torch.zeros((16, 27)).reshape(16, 3, 3, 3)
tensor_x = torch.cat((x, b), 0)#32,3,3,3
#print(tensor_x.shape)

w_global['layer1.0.weight'] = tensor_x

end_svd_time = time.time()

svd_compute_time = end_svd_time - start_svd_time

print(f"svd compute time:{svd_compute_time} s")




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






























