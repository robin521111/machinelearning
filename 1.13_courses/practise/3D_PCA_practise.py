print(__doc__)
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA


iris = datasets.load_iris()
y = iris.target 

fig = plt.figure(1,figsize=(8,6))
