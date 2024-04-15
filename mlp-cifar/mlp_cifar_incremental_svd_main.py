import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, FFD, k_svd, FFD1
from models.test import test_img
import time

# import wandb
#
# wandb.init(project="my-tran-project", entity="zhangpan")
# wandb.config = {
# 	"learning_rate": 0.001,
# 	"epochs": 100,
# 	"batch_size": 128
# }

"""
"""

# def text_create(name):
# 	desktop_path = "E:\Cryption\Federal Learning\FLCodes\mlp-cifar-federated-learning-master"
# 	# 新创建的txt文件的存放路径
# 	full_path = desktop_path + name + '.txt'  # 也可以创建一个.doc的word文档
# 	file = open(full_path, 'w')

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
	"""
	elif args.model == 'cnn' and args.dataset == 'mnist':
	 # 将CNNMnist模型放到设备上，该模型是全局模型
	net_glob = CNNMnist(args=args).to(args.device)
	#mlp即多层感知器
	elif
	"""
	
	# print(net_glob)
	net_glob.train()
	# copy weights
	w_global = net_glob.state_dict()
	# print(w_glob)
	
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
	
	# 获取第二层权重矩阵
	weight_fc2 = net_glob.layer_hidden.weight.detach().cpu().numpy()
	
	# 示例数据
	x_new = torch.randn(10, 200)  # 新数据，与第二层的输出大小相匹配
	
	# 对权重参数进行增量 SVD
	def incremental_svd(param, x_new):
		
		U, Sigma, Vt = np.linalg.svd(param, full_matrices=False)
		
		p = np.dot(x_new, Vt.T)
		new_param = np.hstack((param, p))
		return new_param
	
	
	# 对第二层权重矩阵进行增量 SVD
	updated_weight_fc2 = incremental_svd(weight_fc2, x_new.numpy())
	
	# 更新模型的第二层权重矩阵
	net_glob.layer_hidden.weight.data = torch.from_numpy(updated_weight_fc2)
	
	end_svd_time = time.time()
	
	svd_compute_time = end_svd_time - start_svd_time
	
	print(f"incremental svd compute time:{svd_compute_time} s")
	
	
	
	# # training
	# loss_train1 = []
	# cv_loss, cv_acc = [], []
	# val_loss_pre, counter = 0, 0
	# net_best = None
	# best_loss = None
	# val_acc_list, net_list = [], []
	#
	# if args.all_clients:
	# 	print("Aggregation over all clients")
	# 	w_locals = [w_glob for i in range(args.num_users)]
	#
	# # filename = 'mlp_cifar_svd'
	# # text_create(filename)
	# # output = sys.stdout
	# # outputfile = open("E:\Cryption\Federal Learning\FLCodes\mlp-cifar-federated-learning-master\\" + filename + '.txt',
	# # 'w')
	# # sys.stdout = outputfile
	# start_time = time.time()
	# for iter in range(args.epochs):
	# 	loss_locals = []
	# 	if not args.all_clients:
	# 		w_locals = []
	# 	# args.frac * args.num_users表示选中的用户的数量
	# 	m = max(int(args.frac * args.num_users), 1)
	# 	# np.random.choice(range(args.num_users), m, replace=False)表示从所有的用户中随机选取m个用户
	# 	idxs_users = np.random.choice(range(args.num_users), m, replace=False)
	# 	for idx in idxs_users:
	# 		local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
	# 		w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
	# 		if args.all_clients:
	# 			w_locals[idx] = copy.deepcopy(w)
	# 		else:
	# 			w_locals.append(copy.deepcopy(w))
	# 		loss_locals.append(copy.deepcopy(loss))
	# 	# update global weights即更新全局模型中的权重
	# 	w_glob = FedAvg(w_locals)
	#
	# 	# copy weight to net_glob,上传到全局模型中
	# 	net_glob.load_state_dict(w_glob)
	#
	#
	# 	# print loss
	# 	loss_avg = sum(loss_locals) / len(loss_locals)
	# 	elapsed_1 = (time.time() - start_time)
	#
	# 	# testing
	# 	net_glob.eval()
	# 	acc_train, loss_train = test_img(net_glob, dataset_train, args)
	# 	acc_test, loss_test = test_img(net_glob, dataset_test, args)
	#
	# 	elapsed_2 = (time.time() - start_time - elapsed_1) / 60
	#
	# 	acc_train_1, acc_test_1 = acc_train.numpy(), acc_test.numpy()
	#
	# 	print(f'Training Time (minutes)=: {elapsed_1 / 60:.4f} min')
	# 	print('Train Round {:3d}, Train Average Loss {:.3f}, Train acc {}'.format(iter+1, loss_train, acc_train_1))
	#
	#
	# 	print('Test Round {:3d}, Test Average Loss {:.3f}, Test acc {}'.format(iter+1, loss_test, acc_test_1))
	# 	print(f'Predit Time (minutes)=: {elapsed_2:.4f} min')
	#
	# 	# wandb.log({"Train loss": loss_train, 'epoch': iter + 1})
	# 	# wandb.log({"Test loss": loss_test, 'epoch': iter + 1})
	# 	# wandb.log({"Train acc": acc_train_1, 'epoch': iter + 1})
	# 	# wandb.log({"Test acc": acc_test_1, 'epoch': iter + 1})
	# 	# wandb.log({"Train time": elapsed_1 / 60, 'epoch': iter + 1})
	# 	# wandb.log({"Predit Time": elapsed_2, 'epoch': iter + 1})
	#
	# # print("Training accuracy: {:.2f}".format(acc_train))
	# # print("Testing accuracy: {:.2f}".format(acc_test))
	# end = time.time()
	# elapsed = (end - start_time) / 60
	# print(f"Running time:{elapsed:.2f} min")
