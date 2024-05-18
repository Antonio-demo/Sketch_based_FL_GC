## param setting options:

# epoch = 140
# transforms = flip+colour jitter
# results = 80%

# epoch = 80
# transforms = None
# results = 78%

# epoch = 250
# transforms = resize(40)+flip+rotate(20)+crop(32)+colour jitter
# results = 69%

# epoch = 250
# transforms = flip+rotate(20)+colour jitter
# results = 79%
import time

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

import os
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
from torch.autograd import Variable
import pickle
import datetime
from Fed import clip_grad, gaussian_noise, countsketch, rand_hashing


def progress_bar(progress, size=10, arrow='>'):
	pos = int(min(progress, 1.0) * size)
	return '[{bar:<{size}.{trim}}]'.format(
		bar='{:=>{pos}}'.format(arrow, pos=pos),
		size=size - 1,  # Then final tick is covered by the end of bar.
		trim=min(pos, size - 1))


# Hyper Parameters
ModelName = 'cnn'
EPOCH = 200  # train all the training data n times
BATCH_SIZE = 128  # No. images in one batch
LR = 0.001  # learning rate
Method = 'OnlyCountSketch'
DOWNLOAD_CIFAR10 = False


# Transformation settings for training and testing sets
def train_tf(x):
	im_aug = transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.RandomRotation(20),
		transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
		transforms.ToTensor(),
		transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
	])
	x = im_aug(x)
	return x


def test_tf(x):
	im_aug = transforms.Compose([
		# transforms.Resize(96),
		transforms.ToTensor(),
		transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
	])
	x = im_aug(x)
	return x


# # With no data augmentation
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# CIFAR10 image data set downloads
if not (os.path.exists('./cifar10/')) or not os.listdir('./cifar10/'):
	# not mnist dir or mnist is empyt dir
	DOWNLOAD_CIFAR10 = True

train_data = torchvision.datasets.CIFAR10(
	root='./data/',
	train=True,  # this is training data
	# transform=transform,    # Converts a PIL.Image or numpy.ndarray to
	# torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
	transform=train_tf,
	download=DOWNLOAD_CIFAR10
)

# Data Loader for easy mini-batch return in training
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_data = torchvision.datasets.CIFAR10(
	root='./data/',
	train=False,
	# transform=transform,
	transform=test_tf,
	download=DOWNLOAD_CIFAR10
)

test_loader = Data.DataLoader(dataset=test_data, batch_size=(BATCH_SIZE // 2), shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# =====================================#=====================================#
# construct network #
# =====================================#=====================================#
class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(
				in_channels=3,  # in_channel (in_img height)
				out_channels=16,  # out_channel (output height/No.filters)
				kernel_size=5,  # kernel_size
				stride=1,  # filter step
				padding=2,
			),
			nn.ReLU(),
			nn.Dropout(0.2)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(
				in_channels=16,  # in_channel (in_img height)
				out_channels=16,  # out_channel (output height/No.filters)
				kernel_size=3,  # kernel_size
				stride=1,  # filter step
				padding=1,
			),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.Dropout(0.25)
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(
				in_channels=16,  # in_channel (in_img height)
				out_channels=32,  # out_channel (output height/No.filters)
				kernel_size=5,  # kernel_size
				stride=1,  # filter step
				padding=2,
			),
			nn.ReLU(),
			nn.Dropout(0.5)
		)
		self.conv4 = nn.Sequential(
			nn.Conv2d(
				in_channels=32,  # in_channel (in_img height)
				out_channels=32,  # out_channel (output height/No.filters)
				kernel_size=3,  # kernel_size
				stride=2,  # filter step
				padding=1,
			),
			nn.ReLU(),
			# nn.MaxPool2d(2, 2),
			nn.Dropout(0.5)
		)
		self.fc = nn.Sequential(
			nn.BatchNorm1d(2048),
			nn.Linear(2048, 218),
			# nn.Dropout(0.5),
			# nn.Linear(512, 256),
			# nn.Dropout(0.5),
			nn.ReLU(),
			nn.Linear(218, 10)
		)

	def forward(self, x):
		out = self.conv1(x)
		# print(out.size(1))
		out = self.conv2(out)
		# print(out.size(1))
		out = self.conv3(out)
		# print(out.size(1))
		out = self.conv4(out)
		# print(out.size(1))
		out = out.view(out.size(0), -1)
		out = self.fc(out)
		return out


cnn = CNN()
print(cnn)  # net architecture

# =====================================#=====================================#
# loss func #
# =====================================#=====================================#
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


def validate(loader, name, old_loss=None, old_acc=None):  # validate on the whole dataset(验证整个数据集)
	cnn.eval()  # eval mode (different batchnorm, dropout, etc.)
	with torch.no_grad():
		correct = 0
		loss = 0
		for images, labels in loader:  # everytime load one batch
			# load one batch images (128) in once
			images, labels = Variable(images), Variable(labels)
			# forward to network
			outputs = cnn(images)
			# print(outputs.data.size()) # [128 10] 128-images 10-classes predictions
			_, predicts = torch.max(outputs.data, 1)
			# find out the max prediction corresponding class as results
			# print(predicts.size()) # 128 predicted results for the 128 images in this batch
			correct += (predicts == labels).sum().item()  # check matchness with ground truth
			loss += loss_func(outputs, labels).item()  # accumulated loss through the current batch images
	sign = lambda x: x and (-1, 1)[x > 0]
	compsymb = lambda v: {-1: 'v', 0: '=', 1: '^'}[sign(v)]

	avg_loss = loss / len(loader)  # len(loader) = 391 (num_batch)
	acc = correct / len(loader.dataset)  # len(loader.dataset) = 50000 (total num_img in cifar10)

	print(('[{name} images]'
		   '\t avg loss: {avg_loss:5.3f}{loss_comp}'
		   ', accuracy: {acc:6.2f}%{acc_comp}').format(
		name=name, avg_loss=avg_loss, acc=100 * acc,
		loss_comp='' if old_loss is None else compsymb(avg_loss - old_loss),
		acc_comp='' if old_acc is None else compsymb(acc - old_acc)))
	return avg_loss, acc


# =====================================#=====================================#
# Training #
# =====================================#=====================================#
if __name__ == '__main__':

	print('TRAINING')
	print('=' * 30)
	print('''\
    Epoches: {EPOCH}
    Batch size: {BATCH_SIZE}
    Learning rate: {LR}
    '''.format(**locals()))

	running_loss_size = max(1, len(train_loader) // 10)  # 391 // 10
	train_loss, train_accuracy = None, None
	test_loss, test_accuracy = None, None

	# storing loss during training for plotting
	ts_loss_values = []
	tr_loss_values = []
	ts_acc_values = []
	tr_acc_values = []

	list1 = []

	start = time.perf_counter()
	# global model
	cnn.train()

	# copy weights
	w_global = cnn.state_dict()
	# print(w_global)
	conv1_weight = w_global['conv1.0.weight']
	# 总共1200元素，16*3*5*5
	# print(conv1_weight.shape)
	# print(conv1_weight.numel())
	array_conv1_weight = conv1_weight.numpy().reshape(16, 75)
	# 矩阵每一列的平均数
	raw_avg = np.mean(array_conv1_weight, axis=0)
	for index in range(array_conv1_weight.shape[0]):
		g = array_conv1_weight[index, :] - raw_avg
		list1.append(g)
	# array_list为16*75
	array_list = np.array(list1)
	#将列表转换成张量,为16*75
	tensor_list = torch.from_numpy(array_list)
	#print(tensor_list.shape)
	hash_idx, rand_sgn = rand_hashing(75,3)
	x1 = countsketch(tensor_list, hash_idx, rand_sgn)
	#tensor_list1为16*25，共400个元素
	#print(tensor_list1.shape)
	tensor_zero_matrix = torch.zeros_like(x1)
	# 张量拼接
	stack_x1 = torch.stack((x1, tensor_zero_matrix), dim=0).reshape(2, 16, 5, 5)
	tensor_one_matrix = torch.zeros_like(stack_x1)
	stack_x1_again = torch.stack((stack_x1, tensor_one_matrix), dim=0).reshape(4, 16, 5, 5)
	stack_x1_again = stack_x1_again[0:3].reshape(16, 3, 5, 5)
	#print(stack_x1_again.shape)
	w_global['conv1.0.weight'] = stack_x1_again
	clipgrad = clip_grad(w_global['conv1.0.weight'], 3)
	gaussian_result = gaussian_noise(clipgrad, 2, 2, 1e-5)

	# training and testing
	for epoch in range(EPOCH):

		running_loss = 0.0
		for i, data in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
			# i = nth batch, data = all the data in the nth batch
			# get the inputs
			inputs, labels = data
			# wrap them in Variable
			inputs, labels = Variable(inputs), Variable(labels)
			# clear gradients for this training step
			optimizer.zero_grad()
			# forward + backward + optimize
			# outputs is gradient
			outputs = cnn(inputs)  # cnn output

			loss = loss_func(outputs, labels)  # cross entropy loss
			loss.backward()  # backpropagation, compute gradients
			optimizer.step()  # apply gradients

			running_loss += loss.item()
			# i % 39 == 38 every 39 batches print progress
			if i % running_loss_size == running_loss_size - 1:
				print('[{}] Epoch {} {} loss: {:.3f}'.format(
					datetime.datetime.now().strftime('%H:%M:%S'),
					epoch + 1,
					progress_bar((i + 1) / len(train_loader)),
					running_loss / running_loss_size))
				running_loss = 0.0  # running loss cleared every 39 batches

		cnn.load_state_dict(w_global)
		# after finish one epoch training, validate the whole training and testing dataset
		train_loss, train_accuracy = validate(train_loader, 'train', train_loss, train_accuracy)
		tr_loss_values.append(train_loss)
		tr_acc_values.append(train_accuracy)

		test_loss, test_accuracy = validate(test_loader, 'test', test_loss, test_accuracy)
		ts_loss_values.append(test_loss)
		ts_acc_values.append(test_accuracy)

		end = time.perf_counter()
		print("Running time:%s seconds" % (end - start))

	print('Finished Training')
	# torch.save(cnn, 'cnn.pkl')

	# 将模型的训练的损失值、准确度和测试的损失值、准确度保存下来
	file_name = '/home/test/Desktop/smallNetonCifar10/model/{}_{}_{}_{}_{}' \
		.format(ModelName, EPOCH, BATCH_SIZE, LR, Method)

	with open(file_name, 'wb') as f:
		pickle.dump([train_loss, train_accuracy, test_loss, test_accuracy], f)

	# =====================================#=====================================#
	# Testing #
	# =====================================#=====================================#
	correct = 0
	total = 0
	for data in test_loader:
		images, labels = data
		outputs = cnn(Variable(images))
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum()

	print('Accuracy of the network on the 10000 test images: %d %%' % (
			100 * correct / total))


	# =====================================#=====================================#
	# Plotting #
	# =====================================#=====================================#

	def plot_l_acc(x_axis, ts_l, tr_l, ts_acc, tr_acc):
		n_epoch = np.linspace(0, x_axis, x_axis)
		plt.figure()
		plt.plot(n_epoch, ts_l, color='blue', linewidth=2, label='Test')
		plt.plot(n_epoch, tr_l, color='red', linewidth=2, label='Train')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.title('Loss Vs. Epochs')
		plt.legend(loc='upper right')

		plt.figure()
		plt.plot(n_epoch, ts_acc, color='blue', linewidth=2, label='Test')
		plt.plot(n_epoch, tr_acc, color='red', linewidth=2, label='Train')
		plt.xlabel('No. Epochs')
		plt.ylabel('Accuracy')
		plt.title('Loss Vs. Accuracy')
		plt.legend(loc='upper right')

		plt.show()


	plot_l_acc(EPOCH, ts_loss_values, tr_loss_values, ts_acc_values, tr_acc_values)


"""
	# print(array_list.shape)
	result = FFD(array_list)
	# print(result.shape)
	x = k_svd(result, 16).reshape(16, 75)
	hash_idx, rand_sgn = rand_hashing(75,3)
	countsketch_x = countsketch(x, hash_idx, rand_sgn)
	x1 = countsketch_x.reshape(1,16,5,5)
	#print(countsketch_x.numel())
	#print(countsketch_x.shape)
	tensor_zero_matrix = torch.zeros_like(x1)
	#张量拼接
	stack_x1 = torch.stack((x1, tensor_zero_matrix), dim=0).reshape(2,16,5,5)
	tensor_one_matrix = torch.zeros_like(stack_x1)
	stack_x1_again = torch.stack((stack_x1,tensor_one_matrix), dim=0).reshape(4,16,5,5)
	stack_x1_again = stack_x1_again[0:3].reshape(16,3,5,5)
	#print(stack_x1_again.shape)

	w_global['conv1.0.weight'] = stack_x1_again
	clipgrad = clip_grad(w_global['conv1.0.weight'], 3)
	gaussian_result = gaussian_noise(clipgrad, 2, 2, 1e-5)
"""