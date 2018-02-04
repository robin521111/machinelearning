
# coding: utf-8

# # Set operations on RDDs

# [Introduction to Spark with Python, by Jose A. Dianes](https://github.com/jadianes/spark-py-notebooks)

# Spark supports many of the operations we have in mathematical sets, such as union and intersection, even when the RDDs themselves are not properly sets. It is important to note that these operations require that the RDDs being operated on are of the same type.  

# Set operations are quite straightforward to understand as it work as expected. The only consideration comes from the fact that RDDs are not real sets, and therefore operations such as the union of RDDs doesn't remove duplicates. In this notebook we will have a brief look at `subtract`, `distinct`, and `cartesian`.       

# ## Getting the data and creating the RDD

# As we did in our first notebook, we will use the reduced dataset (10 percent) provided for the KDD Cup 1999, containing nearly half million network interactions. The file is provided as a Gzip file that we will download locally.

# In[1]:
from pyspark import SparkContext
from pyspark import SparkConf
print("Successfully imported Spark Modules")

from operator import add

conf = SparkConf().setMaster("local").setAppName("My App").set("spark.executor.extraJavaOptions", "-Dfile.encoding=UTF-8")
sc=SparkContext("local")
import urllib
#f = urllib.urlretrieve ("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz", "kddcup.data_10_percent.gz")


# In[2]:

#data_file = "./kddcup.data_10_percent.gz"
data_file = "D:/data_and_dep/kddcup.data_10_percent.gz"

raw_data = sc.textFile(data_file)


# ## Getting attack interactions using `subtract`

# For illustrative purposes, imagine we already have our RDD with non attack (normal) interactions from some previous analysis.   

# In[3]:

normal_raw_data = raw_data.filter(lambda x: "normal." in x)


# We can obtain attack interactions by subtracting normal ones from the original unfiltered RDD as follows.  

# In[4]:

attack_raw_data = raw_data.subtract(normal_raw_data)


# Let's do some counts to check our results.  

# In[5]:

from time import time

# count all
t0 = time()
raw_data_count = raw_data.count()
tt = time() - t0
print("All count in {} secs".format(round(tt,3)))


# In[6]:

# count normal
t0 = time()
normal_raw_data_count = normal_raw_data.count()
tt = time() - t0
print("Normal count in {} secs".format(round(tt,3)))


# In[7]:

# count attacks
t0 = time()
attack_raw_data_count = attack_raw_data.count()
tt = time() - t0
print("Attack count in {} secs".format(round(tt,3)))


# In[8]:

print("There are {} normal interactions and {} attacks, from a total of {} interactions".format(normal_raw_data_count,attack_raw_data_count,raw_data_count))


# So now we have two RDDs, one with normal interactions and another one with attacks.  

# ## Protocol and service combinations using `cartesian`

# We can compute the Cartesian product between two RDDs by using the `cartesian` transformation. It returns all possible pairs of elements between two RDDs. In our case we will use it to generate all the possible combinations between service and protocol in our network interactions.  
# 
# First of all we need to isolate each collection of values in two separate RDDs. For that we will use `distinct` on the CSV-parsed dataset. From the [dataset description](http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names) we know that protocol is the second column and service is the third (tag is the last one and not the first as appears in the page).   

# So first, let's get the protocols.  

# In[9]:

csv_data = raw_data.map(lambda x: x.split(","))
protocols = csv_data.map(lambda x: x[1]).distinct()
protocols.collect()


# Now we do the same for services.  

# In[10]:

services = csv_data.map(lambda x: x[2]).distinct()
services.collect()


# A longer list in this case.

# Now we can do the cartesian product.  

# In[11]:

product = protocols.cartesian(services).collect()
print("There are {} combinations of protocol X service".format(len(product)))


# Obviously, for such small RDDs doesn't really make sense to use Spark cartesian product. We could have perfectly collected the values after using `distinct` and do the cartesian product locally. Moreover, `distinct` and `cartesian` are expensive operations so they must be used with care when the operating datasets are large.    
