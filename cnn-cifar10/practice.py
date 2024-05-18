"""
import numpy as np


array_number = [[1,2,3],[4,5,6],[7,8,9], [10,11,12]]
t = np.array(array_number)
print(t)


avg = np.mean(t, axis=0)
print(avg)

g1 = t[0, :] - avg
g2 = t[1, :] - avg
g3 = t[2, :] - avg
#print(t.shape[0])
list1 = []

#矩阵行数
for i in range(t.shape[0]):
	g = t[i , :] - avg
	#print(g)
	list1.append(g)
print("===========================================")
array_list1 = np.array(list1)
print(array_list1)
print(type(array_list1))
"""

import math
import numpy as np
import torch


def countsketch(a, hash_idx, rand_sgn):
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

t = [[1,2,3,1],[4,5,6,4],[7,8,9,7],[10,11,12,10]]
matrix1 = np.array(t)
tensor1 = torch.from_numpy(matrix1)
D = tensor1.repeat(1,1,2)
B = torch.zeros_like(tensor1)

F = torch.stack((tensor1,B), dim=0)
print(F.shape)

print(F)
F = F[0: 1]
print(F)
print(F.shape)








