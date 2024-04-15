# -*- coding: utf-8 -*-
# Python version: 3.6
import time

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms

from utils.options import args_parser
from models.Nets import MLP, CNNMnist, CNNCifar
import numpy as np
from models.Fed import FedAvg, FFD, k_svd

def test(net_g, data_loader):
	# testing
	net_g.eval()
	test_loss = 0
	correct = 0
	l = len(data_loader)
	for idx, (data, target) in enumerate(data_loader):
		data, target = data.to(args.device), target.to(args.device)
		log_probs = net_g(data)
		test_loss += F.cross_entropy(log_probs, target).item()
		y_pred = log_probs.data.max(1, keepdim=True)[1]
		correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
	acc = correct / len(data_loader.dataset)
	test_loss /= len(data_loader.dataset)
	print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
		test_loss, correct, len(data_loader.dataset),
		100. * correct / len(data_loader.dataset)))

	return acc, test_loss

loss_func = nn.CrossEntropyLoss()

def validate(net_g, loader, name, old_loss=None, old_acc=None):  # validate on the whole dataset(验证整个数据集)

	net_g.eval()  # eval mode (different batchnorm, dropout, etc.)
	with torch.no_grad():
		correct = 0
		loss = 0
		for images, labels in loader:  # everytime load one batch
			# load one batch images (128) in once
			images, labels = Variable(images), Variable(labels)
			# forward to network
			outputs = net_g(images)
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


if __name__ == '__main__':

	train_loss, train_accuracy = None, None
	# parse args
	args = args_parser()
	args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

	torch.manual_seed(args.seed)

	# load dataset and split users
	if args.dataset == 'mnist':
		dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
									   transform=transforms.Compose([
										   transforms.ToTensor(),
										   transforms.Normalize((0.1307,), (0.3081,))
									   ]))
		img_size = dataset_train[0][0].shape
	elif args.dataset == 'cifar':
		transform = transforms.Compose(
			[transforms.ToTensor(),
			 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		dataset_train = datasets.CIFAR10('./data/cifar', train=True, transform=transform, target_transform=None,
										 download=True)
		img_size = dataset_train[0][0].shape
	else:
		exit('Error: unrecognized dataset')

	# build model
	if args.model == 'cnn' and args.dataset == 'cifar':
		net_glob = CNNCifar(args=args).to(args.device)
	elif args.model == 'cnn' and args.dataset == 'mnist':
		net_glob = CNNMnist(args=args).to(args.device)
	elif args.model == 'mlp':
		len_in = 1
		for x in img_size:
			len_in *= x
		net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
	else:
		exit('Error: unrecognized model')
	print(net_glob)

	# training
	optimizer = optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
	train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)

	start = time.perf_counter()
	#global train
	net_glob.train()
	w_glob = net_glob.state_dict()
	layer_input_weight = w_glob['layer_input.weight']

	array_layer_input_weight = layer_input_weight.cpu().numpy()

	# raw_avg是行平均向量的矩阵，总共784个
	list1 = []
	raw_avg = np.mean(array_layer_input_weight, axis=0)
	for index in range(array_layer_input_weight.shape[0]):
		g = array_layer_input_weight[index, :] - raw_avg
		list1.append(g)
	# print(g1)
	array_list = np.array(list1)

	b = [[0.00000000] * 784] * 100
	B = np.matrix(b)

	result = FFD(array_list, B)
	x = k_svd(result, 64)
	tensor_x = torch.from_numpy(x)
	w_glob['layer_input.weight'] = tensor_x


	list_loss = []
	list_acc = []
	net_glob.train()
	for epoch in range(args.epochs):
		batch_loss = []
		for batch_idx, (data, target) in enumerate(train_loader):
			data, target = data.to(args.device), target.to(args.device)
			optimizer.zero_grad()
			output = net_glob(data)
			loss = F.cross_entropy(output, target)
			loss.backward()
			optimizer.step()
			if batch_idx % 50 == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, batch_idx * len(data), len(train_loader.dataset),
						   100. * batch_idx / len(train_loader), loss.item()))
			batch_loss.append(loss.item())

		loss_avg = sum(batch_loss) / len(batch_loss)
		net_glob.load_state_dict(w_glob)
		train_loss, train_accuracy = validate(net_glob,train_loader, 'train',train_loss, train_accuracy)
		print('\nTrain loss:', loss_avg)
		list_loss.append(loss_avg)
		list_acc.append(train_accuracy)

		end = time.perf_counter()
		print("Running time:%s seconds" % (end - start))
	print('Finished Training')

	# testing
	if args.dataset == 'mnist':
		dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True,
									  transform=transforms.Compose([
										  transforms.ToTensor(),
										  transforms.Normalize((0.1307,), (0.3081,))
									  ]))
		test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
	elif args.dataset == 'cifar':
		transform = transforms.Compose(
			[transforms.ToTensor(),
			 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		dataset_test = datasets.CIFAR10('./data/cifar', train=False, transform=transform, target_transform=None,
										download=True)
		test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
	else:
		exit('Error: unrecognized dataset')

	print('test on', len(dataset_test), 'samples')
	test_acc, test_loss = test(net_glob, test_loader)
	print("Training accuracy: {:.2f}".format(train_loss))
	print("Testing accuracy: {:.2f}".format(test_acc))
	
	# plot loss and accuracy
	plt.figure()
	plt.plot(range(len(list_loss)), list_loss)
	plt.xlabel('Epochs')
	plt.ylabel('train_loss')
	plt.title('Loss Vs. Epochs')
	

	plt.figure()
	plt.plot(range(len(list_acc)), list_acc)
	plt.xlabel('Epochs')
	plt.ylabel('train_accuray')
	plt.title('Loss Vs. Accuracy')
	plt.legend(loc='upper right')
	plt.show()
""""""