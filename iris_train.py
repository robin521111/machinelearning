from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
iris_feature= iris.data 
iris_target = iris.target 


feature_train,feature_test,target_train,target_test = train_test_split(iris_feature,iris_target,test_size=0.33,random_state=42)


print(target_train)
print(feature_train)