
# coding: utf-8

# # MLlib: Classification with Logistic Regression  

# [Introduction to Spark with Python, by Jose A. Dianes](https://github.com/jadianes/spark-py-notebooks)

# In this notebook we will use Spark's machine learning library [MLlib](https://spark.apache.org/docs/latest/mllib-guide.html) to build a **Logistic Regression** classifier for network attack detection. We will use the complete [KDD Cup 1999](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) datasets in order to test Spark capabilities with large datasets.  

# Additionally, we will introduce two ways of performing **model selection**: by using a correlation matrix and by using hypothesis testing.  

# ## Getting the data and creating the RDD

# As we said, this time we will use the complete dataset provided for the [KDD Cup 1999](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html), containing nearly half million network interactions. The file is provided as a Gzip file that we will download locally.  
from pyspark import SparkContext
from pyspark import SparkConf
print ("Successfully imported Spark Modules")

from operator import add

conf = SparkConf().setMaster("local").setAppName("My App").set("spark.executor.extraJavaOptions", "-Dfile.encoding=UTF-8")
sc=SparkContext("local")

# In[1]:

import urllib
#f = urllib.urlretrieve ("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz", "kddcup.data.gz")


# In[2]:

#data_file = "./kddcup.data.gz"
data_file = "D:/data_and_dep/kddcup.data_10_percent.gz"

raw_data = sc.textFile(data_file)

print("Train data size is {}".format(raw_data.count()))


# The [KDD Cup 1999](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) also provide test data that we will load in a separate RDD.  

# In[3]:

#ft = urllib.urlretrieve("http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz", "corrected.gz")


# In[4]:

#test_data_file = "./corrected.gz"
test_data_file = "D:/data_and_dep/corrected.gz"

test_raw_data = sc.textFile(test_data_file)

print("Test data size is {}".format(test_raw_data.count()))


# ## Labeled Points

# A labeled point is a local vector associated with a label/response. In [MLlib](https://spark.apache.org/docs/latest/mllib-data-types.html#labeled-point), labeled points are used in supervised learning algorithms and they are stored as doubles. For binary classification, a label should be either 0 (negative) or 1 (positive).  

# ### Preparing the training data

# In our case, we are interested in detecting network attacks in general. We don't need to detect which type of attack we are dealing with. Therefore we will tag each network interaction as non attack (i.e. 'normal' tag) or attack (i.e. anything else but 'normal').  

# In[5]:

from pyspark.mllib.regression import LabeledPoint
from numpy import array

def parse_interaction(line):
    line_split = line.split(",")
    # leave_out = [1,2,3,41]
    clean_line_split = line_split[0:1]+line_split[4:41]
    attack = 1.0
    if line_split[41]=='normal.':
        attack = 0.0
    return LabeledPoint(attack, array([float(x) for x in clean_line_split]))

training_data = raw_data.map(parse_interaction)


# ### Preparing the test data

# Similarly, we process our test data file.  

# In[6]:

test_data = test_raw_data.map(parse_interaction)


# ## Detecting network attacks using Logistic Regression

# [Logistic regression](http://en.wikipedia.org/wiki/Logistic_regression) is widely used to predict a binary response. Spark implements [two algorithms](https://spark.apache.org/docs/latest/mllib-linear-methods.html#logistic-regression) to solve logistic regression: mini-batch gradient descent and L-BFGS. L-BFGS is recommended over mini-batch gradient descent for faster convergence.  

# ### Training a classifier

# In[7]:

from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from time import time

# Build the model
t0 = time()
logit_model = LogisticRegressionWithLBFGS.train(training_data)
tt = time() - t0

print("Classifier trained in {} seconds".format(round(tt,3)))


# ### Evaluating the model on new data

# In order to measure the classification error on our test data, we use `map` on the `test_data` RDD and the model to predict each test point class. 

# In[8]:

labels_and_preds = test_data.map(lambda p: (p.label, logit_model.predict(p.features)))


# Classification results are returned in pars, with the actual test label and the predicted one. This is used to calculate the classification error by using `filter` and `count` as follows.

# In[9]:

t0 = time()
test_accuracy = labels_and_preds.filter(lambda v: v[0] == v[1]).count() / float(test_data.count())
tt = time() - t0

print("Prediction made in {} seconds. Test accuracy is {}".format(round(tt,3), round(test_accuracy,4)))


