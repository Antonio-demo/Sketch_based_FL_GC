import csv
import time
import torch
import torchvision
from tqdm import tqdm
import gc
from google_cifar_Fed import rand_hashing, countsketch_2
from Net import GoogLeNet

# import wandb
# wandb.init(project="my-tran-project", entity="zhangpan")
# wandb.config = {
#   "learning_rate": 0.001,
#   "epochs": 100,
#   "batch_size": 128
# }


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
num_epochs = 101  # To decrease the training time of model.
batch_size = 1
num_classes = 10
learning_rate = 0.001

# Data Loader.
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

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
# print(w_global)


print(w_global.keys())
print(len(w_global.keys()))

count = 0

for weight in w_global.keys():
	name = w_global[f'{weight}']
	print(name.shape)
	num_params = name.numel()
	count += num_params

print(count)

start_sketch_time = time.time()

conv_weight = w_global['conv1.conv.weight']
conv_weight = conv_weight.reshape(64, 48)
# print(conv_weight.shape)#64,3,4,4

hash_idx, rand_sgn = rand_hashing(48, 2)
countsketch_x = countsketch_2(conv_weight, hash_idx, rand_sgn)#64,24
#print(countsketch_x.shape)#

b = torch.ones((64,24)).to(device)
tensor_x = torch.cat((countsketch_x, b), 1).reshape(64,3,4,4)
w_global['conv1.conv.weight'] = tensor_x


end_sketch_time = time.time()

sketch_compute_time = end_sketch_time - start_sketch_time

print(f"sketch compute time:{sketch_compute_time} s")



total_step = len(train_loader)
curr_lr = learning_rate

# f1 = open('/home/user/tmp/GoogleNet_CIFAR10/sketch/google_cifar_sketch_train_loss_acc.csv', 'w')
# csv_writer1 = csv.writer(f1)
#
# line1_1 = ['epoch', 'train loss', 'train acc', 'train time']
# csv_writer1.writerow(line1_1)
#
# f2 = open('/home/user/tmp/GoogleNet_CIFAR10/sketch/google_cifar_sketch_test_loss_acc.csv', 'w')
# csv_writer2 = csv.writer(f2)
#
# line2_1 = ['epoch', 'test loss', 'test acc', 'test time']
# csv_writer2.writerow(line2_1)

# start_time = time.time()
#
# for epoch in range(num_epochs):
#
# 	gc.collect()
# 	torch.cuda.empty_cache()
# 	model.train()
# 	total_num = 0
#
# 	for i, (images, labels) in tqdm(enumerate(train_loader, 1)):
# 		images = images.to(device)
# 		labels = labels.to(device)
#
# 		# Forward pass
# 		outputs = model(images)
# 		loss = criterion(outputs, labels)
#
# 		#running_loss += loss.item() * labels.size(0)
#
# 		_, pred = torch.max(outputs, 1)  # 预测最大值所在的位置标签
#
#
# 		accuracy = (pred == labels).float().sum().item()
#
# 		running_acc += accuracy
#
# 		#train_loss = running_loss / len(train_dataset)
#
# 		train_acc = running_acc / len(train_dataset)
#
# 		# Backward and optimize
# 		optimizer.zero_grad()
# 		loss.backward()
# 		optimizer.step()
#
# 	# if (i + 1) % 100 == 0:
# 	#     print('Epoch [{}/{}], Step [{}/{}], Loss {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step,
# 	#                                                             loss.item()))
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
# 	# Decay learning rate
# 	if (epoch + 1) % 20 == 0:
# 		curr_lr /= 3
# 		update_lr(optimizer, curr_lr)
#
# 	eval_loss = 0.0
# 	# Test the mdoel.
# 	model.eval()
# 	with torch.no_grad():
# 		correct = 0
# 		total = 0
# 		for images, labels in test_loader:
# 			images = images.to(device)
# 			labels = labels.to(device)
# 			outputs = model(images)
#
# 			_, predicted = torch.max(outputs.data, 1)
#
# 			loss_1 = criterion(outputs, labels)
# 			#eval_loss += loss.item() * labels.size(0)
#
# 			correct += (predicted == labels).sum().item()
#
# 		#test_loss = eval_loss / len(test_dataset)
# 		test_acc = correct / len(test_dataset)
#
# 	elapsed_2 = (time.time() - start_time - elapsed_1) / 60
# 	print(f'Predit Time (minutes)=: {elapsed_2:.4f} min')
#
# 	print('Epoch [{}/{}],Test Loss: {:.4f}, Test Acc: {:.4f}'.format(epoch + 1, num_epochs, round(float(loss_1.item()), 4), test_acc))
# 	# print('Accuracy of the model on the test images: {}'.format(correct / total))
#
# 	# line2 = [epoch + 1, round(float(loss_1.item()), 4), test_acc, round(float(elapsed_2), 4)]
# 	#
# 	# csv_writer2.writerow(line2)
# 	model.load_state_dict(w_global)
#
# 	# wandb.log({"Train loss": loss.item(), 'epoch': epoch + 1})
# 	# wandb.log({"Train acc": train_acc, 'epoch': epoch + 1})
# 	# wandb.log({"Test loss": round(float(loss_1.item()), 4), 'epoch': epoch + 1})
# 	# wandb.log({"Test acc": test_acc, 'epoch': epoch + 1})
# 	# wandb.log({"Train time": elapsed_1 / 60, 'epoch': epoch + 1})
# 	# wandb.log({"Predit time": elapsed_2, 'epoch': epoch + 1})
#
#
# # elapsed = (time.time() - start_time) / 60
# # print(f'Total Training Time: {elapsed:.2f} min')
# # # save model
# # torch.save(model.state_dict(), './goolenet_cifar_sketch.pt')






























