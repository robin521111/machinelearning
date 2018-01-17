# import numpy as np
# import pandas as pd
#
# from pandas import Series,DataFrame
#
# a=np.array([20,30,40,50,60,79])
# b=np.arange(6)
# print(b)
# c=a-b
#
# print(c)
#
# c=19*np.array([24.232,4.2124,6.3463,7.323,6.234])
#
# print(c)
#
#
# data = DataFrame(np.arange(16).reshape(4,4),index=list('abcd'),columns=list('wvxy'))
#
# print(data)
#
# print(data['w'])
# print(data.w)
# print(data[['w']])
#
# print(data[['w','x']])
#
#
#
# df=DataFrame(np.random.rand(4,5),columns=['A','B','C','D','E'])
# df['Col_sum']=df.apply(lambda x:x.sum(),axis=1)
#
#
# print(df)
#
# df.loc['Row_sum']=df.apply(lambda x:x.sum())
# print(df)
#

import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(-1,1,50)
y=x*x*x +1
plt.plot(x,y)
plt.show()
