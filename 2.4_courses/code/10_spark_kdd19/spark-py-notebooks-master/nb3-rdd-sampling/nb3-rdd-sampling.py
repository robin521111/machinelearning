
# coding: utf-8

# # Sampling RDDs

# [Introduction to Spark with Python, by Jose A. Dianes](https://github.com/jadianes/spark-py-notebooks)

# So far we have introduced RDD creation together with some basic transformations such as `map` and `filter` and some actions such as `count`, `take`, and `collect`.  

# This notebook will show how to sample RDDs. Regarding transformations, `sample` will be introduced since it will be useful in many statistical learning scenarios. Then we will compare results with the `takeSample` action.      

# ## Getting the data and creating the RDD

# In this case we will use the complete dataset provided for the KDD Cup 1999, containing nearly half million network interactions. The file is provided as a Gzip file that we will download locally.

# In[1]:
from pyspark import SparkContext
from pyspark import SparkConf
print("Successfully imported Spark Modules")

from operator import add

conf = SparkConf().setMaster("local").setAppName("My App").set("spark.executor.extraJavaOptions", "-Dfile.encoding=UTF-8")
sc=SparkContext("local")
import urllib
#f = urllib.urlretrieve ("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz", "kddcup.data.gz")


# Now we can use this file to create our RDD.

# In[2]:

#data_file = "./kddcup.data.gz"
data_file = "data/kddcup.data_10_percent.gz"
raw_data = sc.textFile(data_file)


# ## Sampling RDDs   

# In Spark, there are two sampling operations, the transformation `sample` and the action `takeSample`. By using a transformation we can tell Spark to apply successive transformation on a sample of a given RDD. By using an action we retrieve a given sample and we can have it in local memory to be used by any other standard library (e.g. Scikit-learn).  

# ### The `sample` transformation

# The `sample` transformation takes up to three parameters. First is whether the sampling is done with replacement or not. Second is the sample size as a fraction. Finally we can optionally provide a *random seed*.  

# In[3]:

raw_data_sample = raw_data.sample(False, 0.1, 1234)
sample_size = raw_data_sample.count()
total_size = raw_data.count()
print("Sample size is {} of {}".format(sample_size, total_size))


# But the power of sampling as a transformation comes from doing it as part of a sequence of additional transformations. This will show more powerful once we start doing aggregations and key-value pairs operations, and will be specially useful when using Spark's machine learning library MLlib.    

# In the meantime, imagine we want to have an approximation of the proportion of `normal.` interactions in our dataset. We could do this by counting the total number of tags as we did in previous notebooks. However we want a quicker response and we don't need the exact answer but just an approximation. We can do it as follows.   

# In[4]:

from time import time

# transformations to be applied
raw_data_sample_items = raw_data_sample.map(lambda x: x.split(","))
sample_normal_tags = raw_data_sample_items.filter(lambda x: "normal." in x)

# actions + time
t0 = time()
sample_normal_tags_count = sample_normal_tags.count()
tt = time() - t0

sample_normal_ratio = sample_normal_tags_count / float(sample_size)
print("The ratio of 'normal' interactions is {}".format(round(sample_normal_ratio,3)))
print("Count done in {} seconds".format(round(tt,3)))


# Let's compare this with calculating the ratio without sampling.  

# In[5]:

# transformations to be applied
raw_data_items = raw_data.map(lambda x: x.split(","))
normal_tags = raw_data_items.filter(lambda x: "normal." in x)

# actions + time
t0 = time()
normal_tags_count = normal_tags.count()
tt = time() - t0

normal_ratio = normal_tags_count / float(total_size)
print("The ratio of 'normal' interactions is {}".format(round(normal_ratio,3)))
print("Count done in {} seconds".format(round(tt,3)))


# We can see a gain in time. The more transformations we apply after the sampling the bigger this gain. This is because without sampling all the transformations are applied to the complete set of data.  

# ### The `takeSample` action  

# If what we need is to grab a sample of raw data from our RDD into local memory in order to be used by other non-Spark libraries, `takeSample` can be used.  

# The syntax is very similar, but in this case we specify the number of items instead of the sample size as a fraction of the complete data size.  

# In[6]:

t0 = time()
raw_data_sample = raw_data.takeSample(False, 400000, 1234)
normal_data_sample = [x.split(",") for x in raw_data_sample if "normal." in x]
tt = time() - t0

normal_sample_size = len(normal_data_sample)

normal_ratio = normal_sample_size / 400000.0
print("The ratio of 'normal' interactions is {}".format(normal_ratio))
print("Count done in {} seconds".format(round(tt,3)))


# The process was very similar as before. We obtained a sample of about 10 percent of the data, and then filter and split.  
# 
# However, it took longer, even with a slightly smaller sample. The reason is that Spark just distributed the execution of the sampling process. The filtering and splitting of the results were done locally in a single node.  
