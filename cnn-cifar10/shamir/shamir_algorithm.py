from random import randint
from algorithm import G_hash
from miler_rabin import g_p, ModReverse

"""
使用说明
只能加解密64位hash值
加密时，
G_si(S,n,k)     加密数据S,给n个人，至少k能解密
返回share[]  有n项
解密时
shamir_decrypt(Share)   Share为share[]至少k个元素的数组
返回要加密的值
"""


# 输入要保护的数据，生成秘密共享数据n条，构建k元多项式
# 生成的素数为（2**64，2**65）就是16*16位二进制数  hash为 64*16
# 把hash分为4端16*16加密
# 返回的是si是s[1][i]+s[2][i]+s[3][i]+s[4][i]
# 解密是 用k-1个 i,s[1][i] （0<i<k）解出 hash[1]..

def G_si(S, n, k):
	nimi_key = k
	hash = [int(S[0:16], 16), int(S[16:32], 16), int(S[32:48], 16), int(S[48:64], 16)]
	# print("秘密 ：hash值",hash)
	a = [0]
	prime = g_p()
	s = [prime, [], [], [], []]
	# print("prime",prime)
	# 生成k-1个系数  -->  a[1 <---> k-1]  a0是秘密值
	while k - 1:
		temp = randint(2 ** 10, prime)
		if temp not in a:
			a.append(temp)
			k = k - 1
	# print(a)

	# 生成共享密钥  Si   n个点
	# hash_num就是s[]的下标，记录 4组n个点
	# i 就是x的值
	# hash[]就是秘密值
	# j就是x的第几项，x的几次方
	# 最后生成 s[[prime,y1,y2,y3...],[],[],[]]
	for hash_num in range(1, 5):
		for i in range(1, n + 1):
			temp = hash[hash_num - 1]
			for j in range(1, k):
				temp = (temp + a[j] * (i ** j)) % prime
			s[hash_num].append([i, temp])
	share = []
	for i in range(0, n):
		share.append([[s[0], nimi_key], s[1][i], s[2][i], s[3][i], s[4][i]])
	# print("share=",share)
	# share =  [[素数, nimikey],[point1],[point2],point[3],point[4]] * n个key
	return share


def shamir_decrypt(s):
	# s=[[素数, nimikey],[point1],[point2],point[3],point[4]] * n
	# k是多项式系数  a0+ a1*x^1+....a(k-1)*x^(k-1)
	k = s[0][0][1]
	p = s[0][0][0]
	if len(s) < k:
		print("密钥不够,无法解密")
		return

	# 获取横坐标值及对应的s下标 [x,i]的数组
	x = []
	for i in range(len(s)):
		x.append([s[i][1][0], i])
	# print("横坐标有",x)
	hash = ""
	for hash_i in range(1, 5):
		L = 0
		# j对应 x 横坐标 及 存储y值的i下标
		for j in x:
			Lx = 1
			Lx1 = 1
			# 求除了 横坐标为j 的x
			xx = []
			for some in x:
				if some != j:
					xx.append(some)
			# print(xx)
			for i in xx:
				Lx = (-1) * Lx * i[0]
				Lx1 = Lx1 * (j[0] - i[0])
			# Lj 的 分子Lx 分母Lx1
			Lx = (Lx + p) % p
			Lx1 = (Lx1 + p) % p
			# print("Lx,",Lx,"Lx1",Lx1)
			Lx1 = ModReverse(Lx1, p)
			L = (L + s[j[1]][hash_i][1] * Lx * Lx1) % p
		# print(L)
		hash = hash + str(hex(L))[2:]
		# print("长度",len(hash))
	return hash


hash = G_hash("zazal")
print(f"明文：{hash}")
# print(int(hash,16))

share = G_si(hash, 3, 3)
print(f"shamir解密结果：{shamir_decrypt(share)}")

s = [share[0], share[1], share[2]]
print(f"共享份额结果：{shamir_decrypt(s)}")




