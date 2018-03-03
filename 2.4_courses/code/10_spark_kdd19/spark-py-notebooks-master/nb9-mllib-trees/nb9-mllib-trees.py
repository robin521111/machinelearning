# coding: utf-8

# # MLlib: Decision Trees  

# [Introduction to Spark with Python, by Jose A. Dianes](https://github.com/jadianes/spark-py-notebooks)

# In this notebook we will use Spark's machine learning library [MLlib](https://spark.apache.org/docs/latest/mllib-guide.html) to build a **Decision Tree** classifier for network attack detection. We will use the complete [KDD Cup 1999](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) datasets in order to test Spark capabilities with large datasets. 

# Decision trees are a popular machine learning tool in part because they are easy to interpret, handle categorical features, extend to the multiclass classification setting, do not require feature scaling, and are able to capture non-linearities and feature interactions. In this notebook, we will first train a classification tree including every single predictor. Then we will use our results to perform model selection. Once we find out the most important ones (the main splits in the tree) we will build a minimal tree using just three of them (the first two levels of the tree in order to compare performance and accuracy.   

# At the time of processing this notebook, our Spark cluster contains:  
# 
# - Eight nodes, with one of them acting as master and the rest as workers.  
# - Each node contains 8Gb of RAM, with 6Gb being used for each node.  
# - Each node has a 2.4Ghz Intel dual core processor.  
# - Running Apache Spark 1.3.1.  

# ## Getting the data and creating the RDD

# As we said, this time we will use the complete dataset provided for the [KDD Cup 1999](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html), containing nearly half million network interactions. The file is provided as a Gzip file that we will download locally.  

# In[1]:
from pyspark import SparkContext
from pyspark import SparkConf
print ("Successfully imported Spark Modules")

from operator import add

conf = SparkConf().setMaster("local").setAppName("My App").set("spark.executor.extraJavaOptions", "-Dfile.encoding=UTF-8")
sc=SparkContext("local")

import urllib


# In[2]:

#f = urllib.urlretrieve ("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz", "kddcup.data.gz")


# In[3]:

#data_file = "./kddcup.data.gz"
data_file = "data/kddcup.data_10_percent.gzz"
raw_data = sc.textFile(data_file)

print("Train data size is {}".format(raw_data.count()))

# The [KDD Cup 1999](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) also provide test data that we will load in a separate RDD.  

# In[4]:

#ft = urllib.urlretrieve("http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz", "corrected.gz")


# In[5]:

test_data_file = "D:/data_and_dep/corrected.gz"
test_raw_data = sc.textFile(test_data_file)

print("Test data size is {}".format(test_raw_data.count()))


# ## Detecting network attacks using Decision Trees

# In this section we will train a *classification tree* that, as we did with *logistic regression*, will predict if a network interaction is either `normal` or `attack`.  

# Training a classification tree using [MLlib](https://spark.apache.org/docs/latest/mllib-decision-tree.html) requires some parameters:  
# - Training data  
# - Num classes  
# - Categorical features info: a map from column to categorical variables arity. This is optional, although it should increase model accuracy. However it requires that we know the levels in our categorical variables in advance. second we need to parse our data to convert labels to integer values within the arity range.  
# - Impurity metric  
# - Tree maximum depth  
# - And tree maximum number of bins  
# 

# In the next section we will see how to obtain all the labels within a dataset and convert them to numerical factors.  

# ### Preparing the data

# As we said, in order to benefits from trees ability to seamlessly with categorical variables, we need to convert them to numerical factors. But first we need to obtain all the possible levels. We will use *set* transformations on a csv parsed RDD.  

# In[6]:

from pyspark.mllib.regression import LabeledPoint
from numpy import array

csv_data = raw_data.map(lambda x: x.split(","))
test_csv_data = test_raw_data.map(lambda x: x.split(","))

protocols = csv_data.map(lambda x: x[1]).distinct().collect()
services = csv_data.map(lambda x: x[2]).distinct().collect()
flags = csv_data.map(lambda x: x[3]).distinct().collect()


# And now we can use this Python lists in our `create_labeled_point` function. If a factor level is not in the training data, we assign an especial level. Remember that we cannot use testing data for training our model, not even the factor levels. The testing data represents the unknown to us in a real case.     

# In[7]:

def create_labeled_point(line_split):
    # leave_out = [41]
    clean_line_split = line_split[0:41]
    
    # convert protocol to numeric categorical variable
    try: 
        clean_line_split[1] = protocols.index(clean_line_split[1])
    except:
        clean_line_split[1] = len(protocols)
        
    # convert service to numeric categorical variable
    try:
        clean_line_split[2] = services.index(clean_line_split[2])
    except:
        clean_line_split[2] = len(services)
    
    # convert flag to numeric categorical variable
    try:
        clean_line_split[3] = flags.index(clean_line_split[3])
    except:
        clean_line_split[3] = len(flags)
    
    # convert label to binary label
    attack = 1.0
    if line_split[41]=='normal.':
        attack = 0.0
        
    return LabeledPoint(attack, array([float(x) for x in clean_line_split]))

training_data = csv_data.map(create_labeled_point)
test_data = test_csv_data.map(create_labeled_point)


# ### Training a classifier

# We are now ready to train our classification tree. We will keep the `maxDepth` value small. This will lead to smaller accuracy, but we will obtain less splits so later on we can better interpret the tree. In a production system we will try to increase this value in order to find a better accuracy.    

# In[8]:

from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from time import time

# Build the model
t0 = time()
tree_model = DecisionTree.trainClassifier(training_data, numClasses=2, 
                                          categoricalFeaturesInfo={1: len(protocols), 2: len(services), 3: len(flags)},
                                          impurity='gini', maxDepth=4, maxBins=100)
tt = time() - t0

print("Classifier trained in {} seconds".format(round(tt,3)))


# ### Evaluating the model

# In order to measure the classification error on our test data, we use `map` on the `test_data` RDD and the model to predict each test point class. 

# In[9]:

predictions = tree_model.predict(test_data.map(lambda p: p.features))
labels_and_preds = test_data.map(lambda p: p.label).zip(predictions)


# Classification results are returned in pars, with the actual test label and the predicted one. This is used to calculate the classification error by using `filter` and `count` as follows.
# In[10]:

t0 = time()
# https://stackoverflow.com/questions/41346054/invalid-syntax-error-building-decision-tree-with-python-and-spark-churn-predic
#test_accuracy = labels_and_preds.filter(lambda (v, p): v == p).count() / float(test_data.count())
test_accuracy = labels_and_preds.filter(lambda p: p[0] == p[1]).count() / float(test_data.count())
tt = time() - t0

print("Prediction made in {} seconds. Test accuracy is {}".format(round(tt,3), round(test_accuracy,4)))


# *NOTE: the zip transformation doesn't work properly with pySpark 1.2.1. It does in 1.3*

# ### Interpreting the model

# Understanding our tree splits is a great exercise in order to explain our classification labels in terms of predictors and the values they take. Using the `toDebugString` method in our three model we can obtain a lot of information regarding splits, nodes, etc.   

# In[11]:

print("Learned classification tree model:")
print(tree_model.toDebugString())


# For example, a network interaction with the following features (see description [here](http://kdd.ics.uci.edu/databases/kddcup99/task.html)) will be classified as an attack by our model:  
# - `count`, the number of connections to the same host as the current connection in the past two seconds, being greater than 32. 
# - `dst_bytes`, the number of data bytes from destination to source, is 0.  
# - `service` is neither level 0 nor 52.  
# - `logged_in` is false.  
# From our services list we know that:  

# In[12]:

print("Service 0 is {}".format(services[0]))
print("Service 52 is {}".format(services[52]))


# So we can characterise network interactions with more than 32 connections to the same server in the last 2 seconds, transferring zero bytes from destination to source, where service is neither *urp_i* nor *tftp_u*, and not logged in, as network attacks. A similar approach can be used for each tree terminal node.     

# We can see that `count` is the first node split in the tree. Remember that each partition is chosen greedily by selecting the best split from a set of possible splits, in order to maximize the information gain at a tree node (see more [here](https://spark.apache.org/docs/latest/mllib-decision-tree.html#basic-algorithm)). At a second level we find variables `flag` (normal or error status of the connection) and `dst_bytes` (the number of data bytes from destination to source) and so on.    

# This explaining capability of a classification (or regression) tree is one of its main benefits. Understaining data is a key factor to build better models.

# ## Building a minimal model using the three main splits

# So now that we know the main features predicting a network attack, thanks to our classification tree splits, let's use them to build a minimal classification tree with just the main three variables: `count`, `dst_bytes`, and `flag`.  

# We need to define the appropriate function to create labeled points.  

# In[13]:

def create_labeled_point_minimal(line_split):
    # leave_out = [41]
    clean_line_split = line_split[3:4] + line_split[5:6] + line_split[22:23]
    
    # convert flag to numeric categorical variable
    try:
        clean_line_split[0] = flags.index(clean_line_split[0])
    except:
        clean_line_split[0] = len(flags)
    
    # convert label to binary label
    attack = 1.0
    if line_split[41]=='normal.':
        attack = 0.0
        
    return LabeledPoint(attack, array([float(x) for x in clean_line_split]))

training_data_minimal = csv_data.map(create_labeled_point_minimal)
test_data_minimal = test_csv_data.map(create_labeled_point_minimal)


# That we use to train the model.  

# In[14]:

# Build the model
t0 = time()
tree_model_minimal = DecisionTree.trainClassifier(training_data_minimal, numClasses=2, 
                                          categoricalFeaturesInfo={0: len(flags)},
                                          impurity='gini', maxDepth=3, maxBins=32)
tt = time() - t0

print("Classifier trained in {} seconds".format(round(tt,3)))


# Now we can predict on the testing data and calculate accuracy.  

# In[17]:

predictions_minimal = tree_model_minimal.predict(test_data_minimal.map(lambda p: p.features))
labels_and_preds_minimal = test_data_minimal.map(lambda p: p.label).zip(predictions_minimal)


# In[18]:

t0 = time()
#test_accuracy = labels_and_preds_minimal.filter(lambda (v, p): v == p).count() / float(test_data_minimal.count())
test_accuracy = labels_and_preds_minimal.filter(lambda v: v[0] == v[1]).count() / float(test_data_minimal.count())
tt = time() - t0

print("Prediction made in {} seconds. Test accuracy is {}".format(round(tt,3), round(test_accuracy,4)))


# So we have trained a classification tree with just the three most important predictors, in half of the time, and with a not so bad accuracy. In fact, a classification tree is a very good model selection tool!    
