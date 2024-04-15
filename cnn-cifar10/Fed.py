import math
import torch
import numpy as np
import numpy.random as npr
#import scipy.sparse as sparse
# from tabulate import tabulate
import time


def Creat_Hadmard(i=4, j=4):
	temp = i & j
	result = 0
	for step in range(4):
		result += ((temp >> step) & 1)
	if 0 == result % 2:
		sign = 1
	else:
		sign = -1
	return sign


def FFD(g):
	F = np.zeros((16, 75))
	inserted_F = g + F
	S = np.random.randint(0, 2, size=[16, 16])

	Row_Matrix = 16
	Column_Matrix = 16
	Hadmard = np.ones((Row_Matrix, Column_Matrix), dtype=np.float32)
	for i in range(Row_Matrix):
		for j in range(Column_Matrix):
			Hadmard[i][j] = Creat_Hadmard(i, j)
	H = Hadmard

	D = np.diag([1, -1] * 8)
	# Phi = S * H * D
	Phi_v = np.dot(S, H)
	Phi = np.dot(Phi_v, D)
	# print(Phi.shape)

	result = np.dot(Phi, inserted_F).reshape(16, 75)
	return result


def k_svd(X, k):
	"""
	奇异值分解
	:param X: 输入的张量梯度
	:param k: 选取前k个数值
	:return: 奇异值分解后的张量梯度
	"""
	U, Sigma, VT = np.linalg.svd(X)
	indexVec = np.argsort(-Sigma)
	K_index = indexVec[:k]
	U = U[:, K_index]
	S = [[0.0 for i in range(k)] for i in range(k)]
	Sigma = Sigma[K_index]
	for i in range(k):
		S[i][i] = Sigma[i]
	VT = VT[K_index, :]
	X = np.dot(S, VT)
	X = torch.from_numpy(X).reshape(16, 3, 5, 5)
	return X


def clip_grad(grad, clip):
	"""
    Gradient clipping(梯度剪裁)
    """
	g_shape = grad.shape
	grad.flatten()
	grad = grad / np.max((1, float(torch.norm(grad, p=2)) / clip))
	grad.view(g_shape)
	return grad


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


def countsketch(a, hash_idx, rand_sgn):
	"""
	It converts m-by-n matrix to m-by-s matrix(把m*n矩阵转换为m*s矩阵)
	:param a: 输入的张量梯度
	:param hash_idx: (q-by-s Torch Tensor) contain random integer in {0, 1, ..., s-1}
	:param rand_sgn: (n-by-1 Torch Tensor) contain random signs (+1 or -1)
	:return: c: m-by-s sketch (Torch Tensor) (result of count sketch)
	"""
	# 访问a中的shape值,
	# m是行数，n是列数
	m, n = a.shape
	# shape[0]和shape[1]分别代表行和列的长度
	s = hash_idx.shape[1]
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
