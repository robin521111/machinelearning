from sklearn.preprocessing import Binarizer
X= [[1.,-1.,2.],[2.,0.,0.],[0.,1.,-1]]
binarizer = Binarizer().fit(X)
print(binarizer.transform(X))