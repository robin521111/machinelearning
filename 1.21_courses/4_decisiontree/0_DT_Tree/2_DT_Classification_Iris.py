from sklearn.datasets import load_iris
from sklearn import tree
import os
import pydot # need install
print(os.getcwd())
clf = tree.DecisionTreeClassifier()
iris = load_iris()
clf = clf.fit(iris.data, iris.target)
tree.export_graphviz(clf, out_file='4_decisiontree/0_DT_Tree/tree.dot')         
(graph,) = pydot.graph_from_dot_file('4_decisiontree/0_DT_Tree/tree.dot')
graph.write_png('4_decisiontree/0_DT_Tree/tree.png')