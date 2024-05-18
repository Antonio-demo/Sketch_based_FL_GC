# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
import math


def FedAvg(w):
	w_avg = copy.deepcopy(w[0])
	for k in w_avg.keys():
		for i in range(1, len(w)):
			w_avg[k] += w[i][k]
		# torch.div是指w_avg[k]除以len(w)
		w_avg[k] = torch.div(w_avg[k], len(w))
	return w_avg


def Creat_Hadmard(i=4, j=4):
	'哈达玛矩阵的维数是2的幂次方，2^(a) 如(2,4,8,16...)'
	temp = i & j
	result = 0
	for step in range(4):
		result += ((temp >> step) & 1)
	if 0 == result % 2:
		sign = 1
	else:
		sign = -1
	return sign


def FFD(g, B):
	F = np.zeros((16, 147))
	inserted_F = g + F
	S = np.random.randint(0, 2, size=[10, 16])
	D = np.diag([1, -1] * 8)
	Row_Matrix = 16
	Column_Matrix = 16
	Hadmard = np.ones((Row_Matrix, Column_Matrix), dtype=np.float32)
	for i in range(Row_Matrix):
		for j in range(Column_Matrix):
			Hadmard[i][j] = Creat_Hadmard(i, j)
	H = Hadmard
	# Phi = S * H * D
	Phi_v = np.dot(S, H)
	Phi = np.dot(Phi_v, D)

	result1 = np.dot(Phi, inserted_F)
	result1 = result1.repeat(2, axis=0)
	result = result1 + B
	return result


def FFD_svd(g):
	F = np.zeros((128, 2352))
	inserted_F = g
	S = np.random.randint(0, 2, size=[50, 128])
	D = np.diag([1, -1] * 64)
	Row_Matrix = 128
	Column_Matrix = 128
	Hadmard = np.ones((Row_Matrix, Column_Matrix), dtype=np.float32)
	for i in range(Row_Matrix):
		for j in range(Column_Matrix):
			Hadmard[i][j] = Creat_Hadmard(i, j)
	H = Hadmard
	# Phi = S * H * D
	Phi_v = np.dot(S, H)
	Phi = np.dot(Phi_v, D)

	result1 = np.dot(Phi, inserted_F)
	result1 = result1.repeat(2, axis=0)

	return result1


def FFD_svd_sketch(g):
	F = np.zeros((200, 2352))
	inserted_F = g + F
	inserted_F = inserted_F[:128, :]
	S = np.random.randint(0, 2, size=[50, 128])
	D = np.diag([1, -1] * 64)
	Row_Matrix = 128
	Column_Matrix = 128
	Hadmard = np.ones((Row_Matrix, Column_Matrix), dtype=np.float32)
	for i in range(Row_Matrix):
		for j in range(Column_Matrix):
			Hadmard[i][j] = Creat_Hadmard(i, j)
	H = Hadmard
	# Phi = S * H * D
	Phi_v = np.dot(S, H)
	Phi = np.dot(Phi_v, D)

	result1 = np.dot(Phi, inserted_F)
	result1 = result1.repeat(2, axis=0)

	return result1


def k_svd(X, k):
	# 奇异值分解
	U, Sigma, VT = np.linalg.svd(X)  # 已经自动排序了
	# 数据集矩阵 奇异值分解  返回的Sigma 仅为对角线上的值

	indexVec = np.argsort(-Sigma)  # 对奇异值从大到小排序，返回索引

	# 根据求得的分解，取出前k大的奇异值对应的U,Sigma,V
	K_index = indexVec[:k]  # 取出前k最大的特征值的索引

	U = U[:, K_index]  # 从U取出前k大的奇异值的对应(按列取)
	S = [[0.0 for i in range(k)] for i in range(k)]
	Sigma = Sigma[K_index]  # 从Sigma取出前k大的奇异值(按列取)
	for i in range(k):
		S[i][i] = Sigma[i]  # 奇异值list形成矩阵
	VT = VT[K_index, :]  # 从VT取出前k大的奇异值的对应(按行取)
	X = np.dot(S, VT)

	return X


def tight_gaussian(data, s, c2, q, t, delta, epsilon, device=None):
	"""
    Gaussian mechanism -- M. Abadi et al., Deep Learning with Differential Privacy.
    sigma >= c2 * (q sqrt{T log1/δ}) / epsilon
    """
	sigma = c2 * q * np.sqrt(t * np.log(1 / delta)) / epsilon
	sigma *= (s ** 2)
	noise = torch.normal(0, sigma, data.shape).to(device)
	return data + noise


def gaussian_noise(grad, s, epsilon, delta, device=None):
	"""
    Gaussian noise to disturb the gradient matrix
    """
	grad.flatten()
	grad = grad / np.max((1, float(torch.norm(grad, p=2)) / s))
	grad.to(device)

	c = np.sqrt(2 * np.log(1.25 / delta))
	sigma = c * s / epsilon
	noise = torch.normal(0, sigma, grad.shape).to(device)
	return grad + noise


def clip_grad(grad, clip):
	"""
    Gradient clipping(梯度剪裁)
    """
	g_shape = grad.shape
	grad.flatten()
	grad = grad / np.max((1, float(torch.norm(grad, p=2)) / clip))
	grad.view(g_shape)
	return grad


def rand_hashing(n, q):
	"""
	Generate random indices and random signs(产生随机指数和随机符号)
	:param n: (integer) number of items to be hashed(要散列的项目数)
	:param q: (integer) map n items to a table of s=n/q rows(将n个项目映射到一个s=n/q行的表中)
	:return: hash_idx: (q-by-s Torch Tensor) contain random integer in {0, 1, ..., s-1}
	rand_sgn: (n-by-1 Torch Tensor) contain random signs (+1 or -1)
	"""
	# Math.floor()为向下取整,返回小于或等于一个给定数字的最大整数,
	# s表示行数，q表示列数
	s = math.floor(n / q)
	# torch.randperm返回一个0到n-1的数组
	t = torch.randperm(n)
	# hash_idx形成q行s列的矩阵
	hash_idx = t[0:(s * q)].reshape((q, s))
	# torch.randint表示最低值0,最高值2,有n列
	rand_sgn = torch.randint(0, 2, (n,)).float() * 2 - 1
	return hash_idx, rand_sgn


def countsketch(a, hash_idx, rand_sgn):
	"""
		It converts m-by-n matrix to m-by-s matrix(把m*n矩阵转换为m*s矩阵)
		:param a: 输入的张量梯度
		:param hash_idx: (q-by-s Torch Tensor) contain random integer in {0, 1, ..., s-1}
		:param rand_sgn: (n-by-1 Torch Tensor) contain random signs (+1 or -1)
		:return: c: m-by-s sketch (Torch Tensor) (result of count sketch)
		"""
	# c = torch.zeros([m, s], dtype=torch.float32)
	# 矩阵相乘,b为m*n矩阵
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	b = a.mul(rand_sgn.to(device))
	# 得出每一行的和,dim=1,列归一，横向压缩；dim=0,行归一，列向压缩
	# c是1*m矩阵
	c = torch.sum(b[:, hash_idx], dim=1)

	# for h in range(s):
	#     selected = hash_idx[:, h]
	#     c[:, h] = torch.sum(b[:, selected], dim=1)

	return c


def countsketch_2(a, hash_idx, rand_sgn):
	"""
		It converts m-by-n matrix to m-by-s matrix(把m*n矩阵转换为m*s矩阵)
		:param a: 输入的张量梯度
		:param hash_idx: (q-by-s Torch Tensor) contain random integer in {0, 1, ..., s-1}
		:param rand_sgn: (n-by-1 Torch Tensor) contain random signs (+1 or -1)
		:return: c: m-by-s sketch (Torch Tensor) (result of count sketch)
		"""
	# c = torch.zeros([m, s], dtype=torch.float32)
	# 矩阵相乘,b为m*n矩阵
	
	b = a.mul(rand_sgn)
	# 得出每一行的和,dim=1,列归一，横向压缩；dim=0,行归一，列向压缩
	# c是1*m矩阵
	c = torch.sum(b[:, hash_idx], dim=1)

	# for h in range(s):
	#     selected = hash_idx[:, h]
	#     c[:, h] = torch.sum(b[:, selected], dim=1)

	return c
