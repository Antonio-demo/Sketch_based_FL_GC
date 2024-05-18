import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, FFD, k_svd, FFD_svd
from models.test import test_img
import time
import pandas as pd
import openpyxl
from tqdm import tqdm


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
	
	# parse(分析) args
	args = args_parser()
	# args.device表示分配可选设备GPU或者CPU
	args.device = torch.device('cuda:{}'.format(args.gpu)
							   if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

	# load dataset and split users
	if args.dataset == 'mnist':
		# transforms.compose串联多个图片，即[transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]
		# 是列表，遍历
		# transforms.ToTensor()是一个将图片转换为形状为(C,H,W)的tensor格式
		# transforms.Normalize((0.1307,), (0.3081,))是将tensor形式归一化，范围在0~1.0之间
		trans_mnist = transforms.Compose([transforms.Grayscale(num_output_channels=3),
										  transforms.ToTensor(),
										  transforms.Normalize((0.1307,), (0.3081,))])
		# 将数据集MNIST分割为训练集和测试集
		dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
		dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
		# sample users
		# args.iid即args的独立同分布
		if args.iid:
			dict_users = mnist_iid(dataset_train, args.num_users)
		else:
			dict_users = mnist_noniid(dataset_train, args.num_users)

	elif args.dataset == 'cifar':
		trans_cifar = transforms.Compose(
			[transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
		dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
		if args.iid:
			dict_users = cifar_iid(dataset_train, args.num_users)
		else:
			exit('Error: only consider IID setting in CIFAR10')
			dict_users = cifar_noniid(dataset_train, args.num_users)
	else:
		exit('Error: unrecognized dataset')

	img_size = dataset_train[0][0].shape

	# build model
	# cnn即卷积神经网络
	# 将CNNMnist模型放到设备上，该模型是全局模型
	# net_glob = CNNMnist(args=args).to(args.device)
	# mlp即多层感知器
	
	if args.model == 'mlp':
		len_in = 1
		for x in img_size:
			len_in *= x
		net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
	else:
		exit('Error: unrecognized model')
	
	#model = MLP(dim_in=784, dim_hidden=128, dim_out=10).to(args.device)
	
	# print(net_glob)
	net_glob = net_glob.to(args.device)
	
	# copy weights
	w_global = net_glob.state_dict()
	# print(w_glob)
	
	# print(w_global.keys())
	# print(len(w_global.keys()))
	#
	# count = 0
	#
	# for weight in w_global.keys():
	# 	name = w_global[f'{weight}']
	# 	print(name.shape)
	# 	num_params = name.numel()
	# 	count += num_params
	#
	# print(count)
	#net_glob.train()
	
	# 获取第二层权重矩阵
	weight_fc2 = w_global['layer_input.weight'].detach().cpu().numpy()
	#print(f"weight_fc2:{weight_fc2.shape}")#200,2352
	#start_svd_time = time.time()
	# 示例数据
	x_new = torch.ones(200, 2352).detach().cpu().numpy() # 新数据，与第二层的输出大小相匹配
	
	# 对权重参数进行增量 SVD
	def incremental_svd(param, x_new):
		
		U, Sigma, Vt = np.linalg.svd(param, full_matrices=False)
		
		p = np.dot(x_new, Vt.T)
		new_param = np.hstack((param, p))
		# print(param.shape)#10,200
		# print(p.shape)#10,10
		
		return new_param
	
	# 对第二层权重矩阵进行增量 SVD
	updated_weight_fc2 = incremental_svd(weight_fc2, x_new)
	
	updated_weight = updated_weight_fc2[:200, :2352].reshape(200,2352)
	# 更新模型的第二层权重矩阵
	updated_weight_1 = torch.from_numpy(updated_weight)
	w_global['layer_input.weight'] = updated_weight_1
	
	# print(w_global.keys())
	# print(len(w_global.keys()))
	
	#end_svd_time = time.time()
	
	#svd_compute_time = end_svd_time - start_svd_time
	
	#print(f"incremental svd compute time:{svd_compute_time} s")
	
	# training
	loss_train1 = []
	cv_loss, cv_acc = [], []
	val_loss_pre, counter = 0, 0
	net_best = None
	best_loss = None
	val_acc_list, net_list = [], []

	if args.all_clients:
		print("Aggregation over all clients")
		w_locals = [w_global for i in range(args.num_users)]
	
	results_df = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Train Accuracy(%)','Training Time(s)'])
	
	results_df_1 = pd.DataFrame(columns=['Epoch', 'Test Loss', 'Test Accuracy(%)', 'Test Time(s)'])
	

	start_time = time.time()
	
	for iter in tqdm(range(args.epochs)):
		
		loss_locals = []
		if not args.all_clients:
			w_locals = []
		# args.frac * args.num_users表示选中的用户的数量
		m = max(int(args.frac * args.num_users), 1)
		# np.random.choice(range(args.num_users), m, replace=False)表示从所有的用户中随机选取m个用户
		idxs_users = np.random.choice(range(args.num_users), m, replace=False)
		for idx in idxs_users:
			local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
			w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
			if args.all_clients:
				w_locals[idx] = copy.deepcopy(w)
			else:
				w_locals.append(copy.deepcopy(w))
			loss_locals.append(copy.deepcopy(loss))
		# update global weights即更新全局模型中的权重
		w_glob = FedAvg(w_locals)

		# copy weight to net_glob,上传到全局模型中
		net_glob.load_state_dict(w_glob)

		# print loss
		loss_avg = sum(loss_locals) / len(loss_locals)
		elapsed_1 = (time.time() - start_time)

		#print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
		loss_train1.append(loss_avg)
		
		# testing
		net_glob.eval()
		acc_train, loss_train = test_img(net_glob, dataset_train, args)
		acc_test, loss_test = test_img(net_glob, dataset_test, args)
		elapsed_2 = (time.time() - start_time - elapsed_1)

		acc_train_1, acc_test_1 = acc_train.numpy(), acc_test.numpy()

		print('Train Round {:3d}, Train Average Loss {:.3f}, Train acc {}'.format(iter + 1, loss_train, acc_train_1))
		print(f'Training Time (s)=: {elapsed_1} s')

		print('Test Round {:3d}, Test Average Loss {:.3f}, Test acc {}'.format(iter + 1, loss_test, acc_test_1))
		print(f'Predit Time (s)=: {elapsed_2} s')
		
		# 将训练损失值和测试准确度结果添加到DataFrame中
		results_df = results_df.append({'Epoch': iter + 1,
		                                'Train Loss': loss_avg,
		                                'Train Accuracy(%)': acc_train_1,
		                                'Training Time(s)': elapsed_1}, ignore_index=True)
		
		# 将训练损失值和测试准确度结果添加到DataFrame中
		results_df_1 = results_df_1.append({'Epoch': iter + 1,
		                                    'Test Loss': loss_test,
		                                    'Test Accuracy(%)': 100 * acc_test_1,
		                                    'Test Time(s)': elapsed_2}, ignore_index=True)
		
		# 将结果保存到 Excel文件
		with pd.ExcelWriter(f'100_MLP_MNIST_ISVD_fedavg_train_results.xlsx') as writer_1:
			results_df.to_excel(writer_1, index=False, sheet_name='Results')
		
		with pd.ExcelWriter(f'100_MLP_MNIST_ISVD_fedavg_test_results.xlsx') as writer_2:
			results_df_1.to_excel(writer_2, index=False, sheet_name='Results')
		


	end = time.time()
	elapsed = (end - start_time) / 60
	print(f"Running time:{elapsed:.2f} min")


