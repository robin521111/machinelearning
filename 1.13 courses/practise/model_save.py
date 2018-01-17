from sklearn.externals import joblib

from sklearn import svm

x=[]
y=[0,1]

clf=svm.SVC()
clf.fit(X,y)
joblib.dump(clf,"train_model.m")

clf=joblib.load('train_model.m')
clf.predict(X)