"""
主要用于素数检测
f_n_1()函数用来分解n − 1为 2^r*d,奇数为d
Multimod()函数用来快速幂分解
Miller_Rabin(n,k)函数调用前两个函数实现Miller_Rabin算法具体步骤
n代表判别，k代表出错概率
返回True即是素数（结果为素数的出错概率为(1/4)**k）
"""
from random import randint


def f_n_1(n):
	r = 0
	if n % 2 == 0:
		n = n / 2
		r += 1
	return [r, n]


def multimod(a, k, n):  # 快速幂取模
	ans = 1
	while (k != 0):
		if k % 2 == 1:  # 奇数
			ans = (ans % n) * (a % n) % n
		a = (a % n) * (a % n) % n
		k = k // 2  # 整除2
	return ans


def Miller_Rabin(n, iter_num):
	# 2 is prime
	if n == 2:
		return True
	# if n is even or less than 2, then n is not a prime
	if n & 1 == 0 or n < 2:
		return False
	# n-1 = (2^s)m
	m, s = n - 1, 0
	while m & 1 == 0:
		m = m >> 1
		s += 1
	# M-R test
	for _ in range(iter_num):
		b = multimod(randint(2, n - 1), m, n)
		if b == 1 or b == n - 1:
			continue
		for __ in range(s - 1):
			b = multimod(b, 2, n)
			if b == n - 1:
				break
		else:
			return False
	return True


# 生产大素数
def g_p():
	p = randint(2 ** 64, 2 ** 65)
	while Miller_Rabin(p, 8) != True:
		if (p % 2) == 1:
			p += 2
		else:
			p = p + 1
	# p,'出错概率为',1/(4**4))
	return p


# 扩展欧几里得求逆元
def exgcd(a, b):
	if b == 0:
		return 1, 0, a
	else:
		x, y, q = exgcd(b, a % b)
		x, y = y, (x - (a // b) * y)
		return x, y, q


# 扩展欧几里得求逆元
def ModReverse(a, p):
	x, y, q = exgcd(a, p)
	if q != 1:
		raise Exception("No solution.")
	else:
		return (x + p) % p  # 防止负数
