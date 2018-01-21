from sklearn.datasets import load_iris
from sklearn import tree
import os
import pydot
print(os.getcwd())
clf = tree.DecisionTreeClassifier(criterion="entropy")
iris = load_iris()
clf = clf.fit(iris.data, iris.target)


tree.export_graphviz(
    clf, out_file='/Users/robin/Documents/MachineLearning/machinelearning/1.21_courses/4_decisiontree/0_DT_Tree/tree.dot')
(graph,) = pydot.graph_from_dot_file(
    '/Users/robin/Documents/MachineLearning/machinelearning/1.21_courses/4_decisiontree/0_DT_Tree/tree.dot')
graph.write_png(
    '/Users/robin/Documents/MachineLearning/machinelearning/1.21_courses/4_decisiontree/0_DT_Tree/tree.png')
