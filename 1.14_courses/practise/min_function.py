# -*- coding: utf-8 -*-
# cangye@Hotmail.com
"""
不需要引入函数库
需要求解的函数为
f(x, y)=x^2+y^2+1
本代码理解即可
后续课程梯度以及最小值计算有相应库进行完成
"""
print(__doc__)

def f(x, y):
    """
    原函数
    """
    return x**2 + y**2 + 1
def df(x, y):
    """
    计算梯度，一般计算时都直接给出梯度形式，不需要进行符号计算
    """
    return 2 * x, 2 * y

# 给定初始值
x = y = 1

# 定义迭代过程
for itr in range(20):
    dx, dy = df(x, y)
    x = x - dx * 0.1
    y = y - dx * 0.1
    print("f({:.2f},{:.2f})={:.2f})".format(x, y, f(x, y)))


import time
time.sleep(5)