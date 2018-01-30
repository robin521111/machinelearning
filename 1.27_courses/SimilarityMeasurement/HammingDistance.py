#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
汉明距离（Hamming distance）是指
两个等长字符串s1与s2之间的汉明距离定义为
将其中一个变为另外一个所需要作的最小替换次数
"""
print(__doc__)

import scipy.spatial.distance as dst
s1 = ['a', 'e', 'c', 'd']
s2 = ['a', 'c', 'e', 'd']
l = len(s1)
print(dst.hamming(s1, s2)*l)