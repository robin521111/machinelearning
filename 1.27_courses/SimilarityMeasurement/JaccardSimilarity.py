#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
举个例子来说电影基数非常庞大
用户看过的电影只占其中非常小的一部分
如果两个用户都没有看过某一部电影（两个都是 0）
并不能说明两者相似
反而言之，如果两个用户都看过某一部电影（序列中都是 1）
则说明用户有很大的相似度。
在这个例子中，序列中等于 1 所占的权重应该远远大于 0 的权重
这就引出下面要说的杰卡德相似系数（Jaccard similarity）
"""
print(__doc__)

import scipy.spatial.distance as dst
#添加减少0不改变相似度
s1 = [1,1,0,1,0,1,0,0,1,0,0,0,0,0]
s2 = [0,1,1,0,0,0,1,1,1,0,0,0,0,0]
l = len(s1)
print(dst.jaccard(s1, s2))