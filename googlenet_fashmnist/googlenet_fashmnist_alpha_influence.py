import csv
import time
import numpy as np
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from google_fashmnist_Fed import k_svd_3, FFD_2, rand_hashing, countsketch
from googlenet_fashmnist_Net import GoogleNet


import wandb
wandb.init(project="resnet_mnist", entity="zhangpan")
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 128
}





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128

# load dataset
transforms = transforms.Compose([transforms.Resize(96), transforms.ToTensor(), ])

train_dataset = torchvision.datasets.FashionMNIST('data', True, transforms, download=True)

test_dataset = torchvision.datasets.FashionMNIST('data', False, transforms, download=True)

train_loader = DataLoader(train_dataset, batch_size, True)

test_loader = DataLoader(test_dataset, batch_size, True)



# google net
model = GoogleNet(1,10)
model = model.to(device)



# train
lr = 0.001

num_epochs = 101

optimizer = optim.Adam(model.parameters(), lr=lr)

criterion = nn.CrossEntropyLoss()

print('start train on', device)

f1 = open('/home/user/tmp/GoogleNet_CIFAR10/googlenet_fashmnist/alpha_influence/google_fashmnist_alpha_influence_train_loss_acc.csv', 'w')
csv_writer1 = csv.writer(f1)

line1_1 = ['alpha', 'epoch', 'train loss', 'train acc', 'train time']
csv_writer1.writerow(line1_1)

f2 = open('/home/user/tmp/GoogleNet_CIFAR10/googlenet_fashmnist/alpha_influence/google_fashmnist_alpha_influence_test_loss_acc.csv', 'w')
csv_writer2 = csv.writer(f2)

line2_1 = ['alpha', 'epoch', 'test loss', 'test acc', 'test time']
csv_writer2.writerow(line2_1)

start_time_1 = time.time()


for number in range(5, 11, 1):
	
	alpha = number / 10
	
	beta = 1.0
	
	print(f"=================================alpha：{alpha}=========================================")
	start_time = time.time()
	
	w_global = model.state_dict()
	# print(w_global)
	conv_weight = w_global['net.0.weight']  # 64,1,7,7
	# print(conv_weight.shape)
	
	array_conv_weight = conv_weight.cpu().numpy().reshape(64, 49)
	
	list1 = []
	
	raw_avg = np.mean(array_conv_weight, axis=0)
	
	for index in range(array_conv_weight.shape[0]):
		g = array_conv_weight[index, :] - raw_avg
		list1.append(g)
	
	array_list = np.array(list1).reshape(64, 49)
	
	result = FFD_2(array_list)
	x = k_svd_3(result, k=32).reshape(32, 49)  # 32*49
	
	hash_idx, rand_sgn = rand_hashing(49, 2)
	
	countsketch_x = countsketch(x, hash_idx, rand_sgn)  # 32*24
	
	b_1 = torch.zeros((32, 25))
	countsketch_x_2 = torch.cat((countsketch_x, b_1), 1)  # 32*49
	
	countsketch_x_1 = torch.cat((countsketch_x, countsketch_x), 0)  # 64*24
	
	b = torch.zeros((64, 25))
	
	tensor_x = torch.cat((countsketch_x_1, b), 1)  # 64*49
	
	alpha = 0.1
	
	error_accumulate = x - countsketch_x_2  # 32*49
	error_accumulate_1 = torch.cat((error_accumulate, error_accumulate), 0)  # 64*49
	tensor_x_1 = tensor_x + alpha * error_accumulate_1 + beta * alpha * error_accumulate_1
	
	
	w_global['net.0.weight'] = tensor_x_1
	
	for epoch in range(num_epochs):
		
		# train_acc = torch.tensor([.0], dtype=torch.float, device=device)
		
		running_acc = 0.0
		
		for images, labels in tqdm(train_loader):
			
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			
			loss = criterion(outputs, labels)
			_, pred = torch.max(outputs.data, 1)
			
			#num_correct = (pred == labels).sum()
			accuracy = (pred == labels).sum().item()
			
			running_acc += accuracy
			
			train_acc = 100 * running_acc / len(train_loader.dataset)
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		
		print('Epoch [{}/{}], Train Loss: {:.6f}, Train Acc: {:.6f}'.format(epoch + 1, num_epochs, round(float(loss.item()), 4),
		                                                                    train_acc))
		
		elapsed_1 = (time.time() - start_time)
		train_time = elapsed_1 / 60
		print(f'Training Time (minutes)=: {elapsed_1 / 60:.4f} min')
		
		line1 = [alpha, epoch + 1, round(float(loss.item()), 4), train_acc, round(float(train_time), 4)]

		csv_writer1.writerow(line1)
		
		eval_loss = 0.0
		# Test the model
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
		
		elapsed_2 = (time.time() - start_time - elapsed_1) / 60
		print(f'Predit Time (minutes)=: {elapsed_2:.4f} min')
		
		print('Epoch [{}/{}],Test Loss: {:.4f}, Test Acc: {:.4f}'.format(epoch + 1, num_epochs, round(float(loss_1.item()), 4), test_acc))
		# print('Accuracy of the model on the test images: {}'.format(correct / total))

	line2 = [alpha, epoch + 1, round(float(loss_1.item()), 4), test_acc, round(float(elapsed_2), 4)]

	csv_writer2.writerow(line2)

	wandb.log({"Train acc": train_acc, 'alpha': alpha})
	wandb.log({"Test acc": test_acc, 'alpha': alpha})




# save the model
elapsed = (time.time() - start_time_1) / 60
print(f'Total Training Time: {elapsed:.2f} min')
# save model
torch.save(model.state_dict(), './goolenet_cifar_alpha_influence.pt')


























