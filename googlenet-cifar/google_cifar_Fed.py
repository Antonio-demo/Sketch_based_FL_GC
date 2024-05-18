import math
import numpy as np
import torch


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


def FFD(g):
	F = np.zeros((64, 48))#64,49
	inserted_F = g + F
	inserted_F = inserted_F[: 64, :]
	S = np.random.randint(0, 2, size=[64, 64])
	D = np.diag([1, -1] * 32)
	
	Row_Matrix = 64
	Column_Matrix = 64
	Hadmard = np.ones((Row_Matrix, Column_Matrix), dtype=np.float32)
	for i in range(Row_Matrix):
		for j in range(Column_Matrix):
			Hadmard[i][j] = Creat_Hadmard(i, j)
	H = Hadmard
	# Phi = S * H * D
	Phi_v = np.dot(S, H)
	Phi = np.dot(Phi_v, D)
	# print(Phi.shape)
	result1 = np.dot(Phi, inserted_F)
	result = result1
	return result


def FFD_2(g):
	F = np.zeros((64, 48))
	inserted_F = g + F
	inserted_F = inserted_F[: 64, :]
	S = np.random.randint(0, 2, size=[64, 64])
	D = np.diag([1, -1] * 32)
	
	Row_Matrix = 64
	Column_Matrix = 64
	Hadmard = np.ones((Row_Matrix, Column_Matrix), dtype=np.float32)
	for i in range(Row_Matrix):
		for j in range(Column_Matrix):
			Hadmard[i][j] = Creat_Hadmard(i, j)
	H = Hadmard
	# Phi = S * H * D
	Phi_v = np.dot(S, H)
	Phi = np.dot(Phi_v, D)
	# print(Phi.shape)
	result1 = np.dot(Phi, inserted_F)
	result = result1
	return result


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
	X = torch.from_numpy(X).reshape(10,3,4,4)
	return X



def k_svd_2(X, k):
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
	X = torch.from_numpy(X).reshape(10, 3, 4, 4)
	return X


def k_svd_3(X, k):
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
	X = torch.from_numpy(X).reshape(40, 3, 4, 4)
	return X


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
	#m, n = a.shape
	# shape[0]和shape[1]分别代表行和列的长度
	s = hash_idx.shape[1]
	# c = torch.zeros([m, s], dtype=torch.float32)
	# 矩阵相乘,b为m*n矩阵
	#b = a.mul(rand_sgn)
	#cuda = torch.cuda.is_available()
	device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
	b = a.mul(rand_sgn)
	
	#print(rand_sgn.is_cuda)
	#print(b.is_cuda)
	# 得出每一行的和,dim=1,列归一，横向压缩；dim=0,行归一，列向压缩
	# c是1*m矩阵
	c = torch.sum(b[:, hash_idx], dim=1)
	
	# for h in range(s):
	#     selected = hash_idx[:, h]
	#     c[:, h] = torch.sum(b[:, selected], dim=1)
	t = b[:, hash_idx]
	print(f"b[:, hash_idx]:{t.shape}")
	print(f"b's shape:{b.shape}")  # 64*48
	print(f"hash idx:{hash_idx.shape}")  # 2*24
	return c


def countsketch_2(a, hash_idx, rand_sgn):
	"""
		It converts m-by-n matrix to m-by-s matrix(把m*n矩阵转换为m*s矩阵)
		:param a: 输入的张量梯度
		:param hash_idx: (q-by-s Torch Tensor) contain random integer in {0, 1, ..., s-1}
		:param rand_sgn: (n-by-1 Torch Tensor) contain random signs (+1 or -1)
		:return: c: m-by-s sketch (Torch Tensor) (result of count sketch)
		"""
	# 矩阵相乘,b为m*n矩阵
	# b = a.mul(rand_sgn)
	# cuda = torch.cuda.is_available()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	b = a.mul(rand_sgn.to(device))
	# print(rand_sgn.is_cuda)
	# print(b.is_cuda)
	# 得出每一行的和,dim=1,列归一，横向压缩；dim=0,行归一，列向压缩
	
	c = torch.sum(b[:, hash_idx], dim=1)#64*24
	#t = b[:, hash_idx]#64,2,24
	print(f"b's shape:{b.shape}")#64*48
	print(f"hash idx:{hash_idx.shape}")#2*24
	
	# print("t：",t.shape)
	# print("c:", c.shape)
	
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
	# torch.randint表示最低值0,最高值2,有n行
	rand_sgn = torch.randint(0, 2, (n,)).float() * 2 - 1
	#print("rand_sgn:",rand_sgn.shape)
	
	return hash_idx, rand_sgn


if __name__ == '__main__':
	
	hash_idx, rand_sgn = rand_hashing(48, 2)
	print(f"hash idx:{hash_idx.shape}")
	
	print(f"rand sgn:{rand_sgn.shape}")
	
	
	result = torch.ones([64,48])
	output = countsketch(result,hash_idx, rand_sgn)
	print(f"output shape:{output.shape}")
	
	








