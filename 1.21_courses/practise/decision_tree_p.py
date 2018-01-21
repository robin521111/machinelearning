from sklearn import tree
from sklearn.datasets import load_iris
import  pydot
iris = load_iris()

data_x = iris.data
data_y = iris.target


clf = tree.DecisionTreeClassifier()

clf.fit(data_x, data_y)

tree.export_graphviz(
    clf, out_file="/Users/robin/Documents/MachineLearning/machinelearning/1.21_courses/4_decisiontree/0_DT_Tree/tree1.dot")
(graph,) = pydot.graph_from_dot_file(
    '/Users/robin/Documents/MachineLearning/machinelearning/1.21_courses/4_decisiontree/0_DT_Tree/tree1.dot')
graph.write_png(
    '/Users/robin/Documents/MachineLearning/machinelearning/1.21_courses/4_decisiontree/0_DT_Tree/tree1.png')
