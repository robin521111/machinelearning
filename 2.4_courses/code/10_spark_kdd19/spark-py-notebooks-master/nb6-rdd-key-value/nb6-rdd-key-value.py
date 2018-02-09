
# coding: utf-8

# # Working with key/value pair RDDs

# [Introduction to Spark with Python, by Jose A. Dianes](https://github.com/jadianes/spark-py-notebooks)

# Spark provides specific functions to deal with RDDs which elements are key/value pairs. They are usually used to perform aggregations and other processings by key.  

# In this notebook we will show how, by working with key/value pairs, we can process our network interactions dataset in a more practical and powerful way than that used in previous notebooks. Key/value pair aggregations will show to be particularly effective when trying to explore each type of tag in our network attacks, in an individual way.  

# ## Getting the data and creating the RDD

# As we did in our first notebook, we will use the reduced dataset (10 percent) provided for the [KDD Cup 1999](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html), containing nearly half million network interactions. The file is provided as a Gzip file that we will download locally.  

# In[1]:
from pyspark import SparkContext
from pyspark import SparkConf
print("Successfully imported Spark Modules")

from operator import add

conf = SparkConf().setMaster("local").setAppName("My App").set("spark.executor.extraJavaOptions", "-Dfile.encoding=UTF-8")
sc=SparkContext("local")

import urllib
#f = urllib.urlretrieve ("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz", "kddcup.data_10_percent.gz")


# In[1]:

#data_file = "./kddcup.data_10_percent.gz"
raw_data = "D:/data_and_dep/kddcup.data_10_percent.gz"

raw_data = sc.textFile(data_file)


# ## Creating a pair RDD for interaction types

# In this notebook we want to do some exploratory data analysis on our network interactions dataset. More concretely we want to profile each network interaction type in terms of some of its variables such as duration. In order to do so, we first need to create the RDD suitable for that, where each interaction is parsed as a CSV row representing the value, and is put together with its corresponding tag as a key.  

# Normally we create key/value pair RDDs by applying a function using `map` to the original data. This function returns the corresponding pair for a given RDD element. We can proceed as follows.  

# In[2]:

csv_data = raw_data.map(lambda x: x.split(","))
key_value_data = csv_data.map(lambda x: (x[41], x)) # x[41] contains the network interaction tag


# We have now our key/value pair data ready to be used. Let's get the first element in order to see how it looks like.  

# In[3]:

key_value_data.take(1)


# ## Data aggregations with key/value pair RDDs

# We can use all the transformations and actions available for normal RDDs with key/value pair RDDs. We just need to make the functions work with pair elements. Additionally, Spark provides specific functions to work with RDDs containing pair elements. They are very similar to those available for general RDDs.  

# For example, we have a `reduceByKey` transformation that we can use as follows to calculate the total duration of each network interaction type.  

# In[4]:

key_value_duration = csv_data.map(lambda x: (x[41], float(x[0]))) 
durations_by_key = key_value_duration.reduceByKey(lambda x, y: x + y)

durations_by_key.collect()


# We have a specific counting action for key/value pairs.  

# In[5]:

counts_by_key = key_value_data.countByKey()
counts_by_key
