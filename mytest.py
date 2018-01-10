import tensorflow as tf 
import numpy as np


# np.__config__.show()


print(np.__version__)

# z[1,2]=1
# print(z)


# print(np.eye(3,3,0))

a = np.array([[[1, 2, 3],[1, 2, 3],[1, 2, 3]],[[1, 2, 3],[1, 2, 3],[1, 2, 3]],[[1, 2, 3],[1, 2, 3],[1, 2, 3]]])
# print(a)
# print(a.T)
# print(a.dtype)

print(a.imag)
print(a.real)
print(a.size)
print(a.itemsize)
print(a.ndim)
print(a.shape)
print(a.strides)
print(np.arange(10).reshape(2,5))
print(np.ravel(a,order='C'))

a=np.ones((1,2,3))
print(a)

print('//move axis')
np.moveaxis(a,0,-2)
print(a)

a=np.arange(4).reshape(2,2)
np.transpose(a)
print(a)
print('array test')
np.asarray(a)
print(a)

np.asmatrix(a)

print(a)

print(np.rad2deg(np.pi))
