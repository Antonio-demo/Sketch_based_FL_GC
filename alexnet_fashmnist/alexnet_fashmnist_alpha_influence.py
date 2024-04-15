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
from alexnet_fashmnist_Fed import k_svd_3, FFD_2, rand_hashing, countsketch

import wandb

wandb.init(project="resnet_mnist", entity="zhangpan")
wandb.config = {
	"learning_rate": 0.001,
	"epochs": 100,
	"batch_size": 128
}

train_dataset = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor())

test_dataset = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor())

batch_size = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size)

test_loader = DataLoader(test_dataset, batch_size=batch_size)

device_1 = "cuda" if torch.cuda.is_available() else "cpu"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using {} device".format(device_1))

f1 = open(
	'/home/user/tmp/GoogleNet_CIFAR10/alexnet_fashmnist/alpha_influence/alexnet_fashmnist_alpha_error_train_loss_acc.csv',
	'w')
csv_writer1 = csv.writer(f1)

line1_1 = ['alpha', 'epoch', 'train loss', 'train acc', 'train time']
csv_writer1.writerow(line1_1)

f2 = open('/home/user/tmp/GoogleNet_CIFAR10/alexnet_fashmnist/alpha_influence/alexnet_fashmnist_alpha_error_test_loss_acc.csv',
	'w')
csv_writer2 = csv.writer(f2)

line2_1 = ['alpha', 'epoch', 'test loss', 'test acc', 'test time']
csv_writer2.writerow(line2_1)

model = AlexNet(num_classes=10, init_weights=True).to(device)
# print(model)
w_global = model.state_dict()
# print(summary(model))
# print(w_global)



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 101

print('Training Started')

start_time_1 = time.time()

for number in range(1, 11, 1):
	
	start_time = time.time()
	
	alpha = number / 10
	
	beta = 1.0
	
	print(f"=================================alpha：{alpha}=========================================")
	model.train()
	
	
	conv_weight = w_global['features.0.weight']  # 32*1*3*3
	# print(conv_weight.shape)
	
	array_conv_weight = conv_weight.cpu().numpy().reshape(16, 18)
	
	list1 = []
	raw_avg = np.mean(array_conv_weight, axis=0)
	
	for index in range(array_conv_weight.shape[0]):
		g = array_conv_weight[index, :] - raw_avg
		list1.append(g)
	
	array_list = np.array(list1).reshape(32, 9)  #
	
	result = FFD_2(array_list)
	
	# print(result.shape)#
	
	x = k_svd_3(result, k=9).reshape(9, 9)  # 9*1*3*3
	
	hash_idx, rand_sgn = rand_hashing(9, 2)
	
	countsketch_x = countsketch(x, hash_idx, rand_sgn)  # 9*4
	
	b = torch.zeros((9, 5))
	countsketch_x_1 = torch.cat((countsketch_x, b), 1)  # 9*9
	
	tensor_x = torch.cat((countsketch_x, countsketch_x), 1)  # 9*8
	
	tensor_x_1 = torch.cat((tensor_x, tensor_x), 1)  # 9*16
	
	b_2 = torch.zeros((9, 16))
	
	tensor_x_2 = torch.cat((tensor_x_1, b_2), 1).reshape(32, 1, 3, 3)  # 9*32
	
	error_accumulate = x - countsketch_x_1  # 9*9
	
	error_accumulate_1 = torch.cat((error_accumulate, x), 0)  # 18*9
	
	error_accumulate_2 = torch.cat((error_accumulate_1, error_accumulate), 0)  # 27*9
	
	b_1 = torch.zeros((5, 9))
	
	error_accumulate_3 = torch.cat((error_accumulate_2, b_1), 0).reshape(32, 1, 3, 3)  # 32*9
	
	
	tensor_x_3 = tensor_x_2 + alpha * error_accumulate_3 + beta * alpha * error_accumulate_3 # 32*9
	
	w_global['layer1.0.weight'] = tensor_x_3
	
	for epoch in range(num_epochs):
		
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
			
			train_acc = 100 * running_acc / (len(train_loader.dataset))
			
			# initial gradient
			optimizer.zero_grad()
			
			# backward propagation
			loss.backward()
			
			# optimise
			optimizer.step()
		
		
		print('Epoch [{}/{}], Train Loss: {:.6f}, Train Acc: {:.6f}'.format(epoch + 1, num_epochs,
		                                                                    round(float(loss.item()), 4), train_acc))
		
		elapsed_1 = (time.time() - start_time)
		train_time = elapsed_1 / 60
		
		print(f'Training Time (minutes)=: {elapsed_1 / 60:.4f} min')
		
		line1 = [alpha, epoch + 1, round(float(loss.item()), 4), train_acc, round(float(train_time), 4)]
		
		csv_writer1.writerow(line1)
		
		# evaluate the model
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
		
		elapsed_2 = (time.time() - start_time - elapsed_1) / 60
		print(f'Predit Time (minutes)=: {elapsed_2:.4f} min')
		
		line2 = [alpha, epoch + 1, round(float(loss_1.item()), 4), test_acc, round(float(elapsed_2), 4)]
		
		csv_writer2.writerow(line2)
		
	wandb.log({"Train acc": train_acc, 'alpha': alpha})
	wandb.log({"Test acc": test_acc, 'alpha': alpha})
	


print('Training Finished')

elapsed = (time.time() - start_time_1) / 60
print(f'Total Training Time: {elapsed:.2f} min')

# Save Models
torch.save(model.state_dict(), "alexnet_fashmnist_alpha_influence.pt")
# print("Saved PyTorch Model State to AlexNet-FashionMnist.pth")

"""
"""





























































