import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, k_svd,FFD_svd_sketch, countsketch_2, rand_hashing
from models.test import test_img
import time
import pandas as pd
import openpyxl
from tqdm import tqdm




if __name__ == '__main__':
	
	# parse(分析) args
	args = args_parser()
	# args.device表示分配可选设备GPU或者CPU
	args.device = torch.device('cuda:{}'.format(args.gpu)
	                           if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
	
	if args.dataset == 'cifar':
		trans_cifar = transforms.Compose([
			transforms.Grayscale(num_output_channels=3),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
		dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
		# if args.iid:
		dict_users = cifar_iid(dataset_train, args.num_users)
		# else:
		# exit('Error: only consider IID setting in CIFAR10')
	else:
		exit('Error: unrecognized dataset')
	img_size = dataset_train[0][0].shape
	
	# build model
	# cnn即卷积神经网络
	# args.model == 'mlp'
	# args.dataset == 'cifar'
	if args.model == 'mlp':
		len_in = 1
		for x in img_size:
			len_in *= x
		net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
	else:
		exit('Error: unrecognized model')
	
	# 将CNNCifar模型放到设备上，该模型是全局模型
	# net_glob = CNNCifar(args=args).to(args.device)
	
	# print(net_glob)
	net_glob.train()
	# copy weights
	w_global = net_glob.state_dict()
	# print(w_glob)
	
	
	# 获取第二层权重矩阵
	weight_fc2 = w_global['layer_input.weight'].detach().cpu().numpy()
	#print(f"weight_fc2:{weight_fc2.shape}")
	# 示例数据
	x_new = torch.ones(200, 3072).detach().cpu().numpy()  # 新数据，与第二层的输出大小相匹配
	
	# 对权重参数进行增量 SVD
	def incremental_svd(param, x_new):
		
		U, Sigma, Vt = np.linalg.svd(param, full_matrices=False)
		
		p = np.dot(x_new, Vt.T)
		new_param = np.hstack((param, p))
		return new_param
	
	# 对第二层权重矩阵进行增量 SVD
	updated_weight_fc2 = incremental_svd(weight_fc2, x_new)
	
	updated_weight = updated_weight_fc2[:200, :3072].reshape(200,3072)
	# 更新模型的第二层权重矩阵
	list1 = []
	# raw_avg是行平均向量的矩阵，总共784个
	raw_avg = np.mean(updated_weight, axis=0)
	for index in range(updated_weight.shape[0]):
		g = updated_weight[index, :] - raw_avg
		list1.append(g)
	# array_list为16*75
	array_list = np.array(list1)
	# print(array_list.shape)
	# 200*3072
	result = FFD_svd_sketch(array_list)
	# 100*3072
	# print(result.shape)
	x = k_svd(result, 50).reshape(50, 3072)
	tensor_x = torch.from_numpy(x)
	# print(tensor_x.shape)#50, 1024
	
	x_svd = x[:, :1024]
	tensor_x_svd = torch.from_numpy(x_svd)
	# print(tensor_x_svd.shape)
	
	hash_idx, rand_sgn = rand_hashing(3072, 3)
	# countsketch_x为50*1024
	countsketch_x = countsketch_2(tensor_x, hash_idx, rand_sgn)
	# print(countsketch_x.shape)#50, 341
	
	error_accumulate = tensor_x_svd - countsketch_x
	
	alpha = 0.1
	countsketch_x = countsketch_x + alpha * error_accumulate
	
	w_global['layer_input.weight'] = countsketch_x
	# print(w_glob['layer_input.weight'].shape)
	
	
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
	
	results_df = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Train Accuracy(%)', 'Training Time(s)'])
	
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
		
		# testing
		net_glob.eval()
		acc_train, loss_train = test_img(net_glob, dataset_train, args)
		acc_test, loss_test = test_img(net_glob, dataset_test, args)

		elapsed_2 = (time.time() - start_time - elapsed_1)

		acc_train_1, acc_test_1 = acc_train.numpy(), acc_test.numpy()

		print(f'Training Time (s)=: {elapsed_1} s')
		print('Train Round {:3d}, Train Average Loss {:.3f}, Train acc {}'.format(iter+1, loss_train, acc_train_1))


		print('Test Round {:3d}, Test Average Loss {:.3f}, Test acc {}'.format(iter+1, loss_test, acc_test_1))
		print(f'Predit Time (s)=: {elapsed_2} s')
		
		# 将训练损失值和测试准确度结果添加到DataFrame中
		results_df = results_df.append({'Epoch': iter + 1,
		                                'Train Loss': loss_train,
		                                'Train Accuracy(%)': 100 * acc_train_1,
		                                'Training Time(s)': elapsed_1}, ignore_index=True)
		
		# 将训练损失值和测试准确度结果添加到DataFrame中
		results_df_1 = results_df_1.append({'Epoch': iter + 1,
		                                    'Test Loss': loss_test,
		                                    'Test Accuracy(%)': 100 * acc_test_1,
		                                    'Test Time(s)': elapsed_2}, ignore_index=True)
		
		# 将结果保存到 Excel文件
		with pd.ExcelWriter('100_mlp_cifar_ISVD_sketch_error_fedavg_train_results.xlsx') as writer_1:
			results_df.to_excel(writer_1, index=False, sheet_name='Results')
		
		with pd.ExcelWriter('100_mlp_cifar_ISVD_sketch_error_fedavg_test_results.xlsx') as writer_2:
			results_df_1.to_excel(writer_2, index=False, sheet_name='Results')
		
		
	end = time.time()
	elapsed = (end - start_time) / 60
	print(f"Running time:{elapsed:.2f} min")
