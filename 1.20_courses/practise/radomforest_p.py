from sklearn.ensemble import AdaBoostClassifier

from sklearn.decomposition import PCA
from  sklearn.tree 
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.linear_model import LogisticRegression



train = np.load(
    "/Users/robin/Documents/MachineLearning/machinelearning/1.20_courses/data/train.npz")
test = np.load(
    "/Users/robin/Documents/MachineLearning/machinelearning/1.20_courses/data/test.npz")
pca = PCA(n_components=30)
X_r = pca.fit(train["images"]).transform(train["images"])
vect_t = np.array([[itr] for itr in range(10)])
X_train = X_r[:6000]
y_train = np.dot(train["labels"][:6000], vect_t).ravel().astype("int")
X_test = X_r[6000:12000]
y_test = np.dot(train["labels"][6000:12000], vect_t).ravel().astype("int")

estimator = AdaBoostClassifier(n_estimators=50)
estimator.fit(X_train,y_train)
predict = estimator.predict(X_test)
print(np.sum(predict == y_test)/6000.)
