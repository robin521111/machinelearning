
# coding: utf-8

# # RDD creation

# #### [Introduction to Spark with Python, by Jose A. Dianes](https://github.com/jadianes/spark-py-notebooks)

# In this notebook we will introduce two different ways of getting data into the basic Spark data structure, the **Resilient Distributed Dataset** or **RDD**. An RDD is a distributed collection of elements. All work in Spark is expressed as either creating new RDDs, transforming existing RDDs, or calling actions on RDDs to compute a result. Spark automatically distributes the data contained in RDDs across your cluster and parallelizes the operations you perform on them.

# #### References

# The reference book for these and other Spark related topics is *Learning Spark* by Holden Karau, Andy Konwinski, Patrick Wendell, and Matei Zaharia.  

# The KDD Cup 1999 competition dataset is described in detail [here](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99).

# ## Getting the data files  

# In this notebook we will use the reduced dataset (10 percent) provided for the KDD Cup 1999, containing nearly half million network interactions. The file is provided as a *Gzip* file that we will download locally.  
from pyspark import SparkContext
from pyspark import SparkConf
print("Successfully imported Spark Modules")

from operator import add

conf = SparkConf().setMaster("local").setAppName("My App").set("spark.executor.extraJavaOptions", "-Dfile.encoding=UTF-8")
sc=SparkContext("local")
# In[31]:

import urllib.request
#f = urllib.request.urlretrieve ("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz", "kddcup.data_10_percent.gz")


# ## Creating a RDD from a file  

# The most common way of creating an RDD is to load it from a file. Notice that Spark's `textFile` can handle compressed files directly.    

# In[32]:

data_file = "D:/data_and_dep/kddcup.data_10_percent.gz"
raw_data = sc.textFile(data_file)


# Now we have our data fileclera loaded into the `raw_data` RDD.

# Without getting into Spark *transformations* and *actions*, the most basic thing we can do to check that we got our RDD contents right is to `count()` the number of lines loaded from the file into the RDD.  

# In[33]:

print(raw_data.count())


# We can also check the first few entries in our data.  

# In[34]:

print(raw_data.take(5))


# In the following notebooks, we will use this raw data to learn about the different Spark transformations and actions.  

# ## Creating and RDD using `parallelize`

# Another way of creating an RDD is to parallelize an already existing list.  

# In[35]:

a = range(100)

data = sc.parallelize(a)


# As we did before, we can `count()` the number of elements in the RDD.

# In[36]:

print(data.count())


# As before, we can access the first few elements on our RDD.  

# In[37]:

print(data.take(5))
