#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
编辑距离是指两个字串之间
由一个转成另一个所需的最少编辑操作次数
许可的编辑操作包括将一个字符替换成另一个字符
插入一个字符，删除一个字符。
"""
print(__doc__)

def EditDist(sm,sn):
    m,n = len(sm)+1,len(sn)+1
    # create a matrix (m*n)
    matrix = [[0]*n for i in range(m)]
    matrix[0][0]=0
    for i in range(1,m):
        matrix[i][0] = matrix[i-1][0] + 1
    for j in range(1,n):
        matrix[0][j] = matrix[0][j-1]+1
    cost = 0
    for i in range(1,m):
        for j in range(1,n):
            if sm[i-1]==sn[j-1]:
                cost = 0
            else:
                cost = 1
            matrix[i][j]=min(matrix[i-1][j]+1,matrix[i][j-1]+1,matrix[i-1][j-1]+cost)
    return matrix[m-1][n-1]

s1 = ['a', 'e', 'c', 'd']
s2 = ['a', 'e', 'd']

print(EditDist(s1, s2))