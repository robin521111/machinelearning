
# coding: utf-8

# # RDD basics

# #### [Introduction to Spark with Python, by Jose A. Dianes](https://github.com/jadianes/spark-py-notebooks)

# This notebook will introduce three basic but essential Spark operations. Two of them are the *transformations* `map` and `filter`. The other is the *action* `collect`. At the same time we will introduce the concept of *persistence* in Spark.    

# ## Getting the data and creating the RDD

# As we did in our first notebook, we will use the reduced dataset (10 percent) provided for the KDD Cup 1999, containing nearly half million network interactions. The file is provided as a Gzip file that we will download locally.

# In[ ]:
from pyspark import SparkContext
from pyspark import SparkConf
print("Successfully imported Spark Modules")

from operator import add

conf = SparkConf().setMaster("local").setAppName("My App").set("spark.executor.extraJavaOptions", "-Dfile.encoding=UTF-8")
sc=SparkContext("local")

import urllib
#f = urllib.urlretrieve ("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz", "kddcup.data_10_percent.gz")


# Now we can use this file to create our RDD.

# In[1]:
data_file = "D:/data_and_dep/kddcup.data_10_percent.gz"
#data_file = "./kddcup.data_10_percent.gz"
raw_data = sc.textFile(data_file)


# ## The `filter` transformation

# This transformation can be applied to RDDs in order to keep just elements that satisfy a certain condition. More concretely, a function is evaluated on every element in the original RDD. The new resulting RDD will contain just those elements that make the function return `True`.

# For example, imagine we want to count how many `normal.` interactions we have in our dataset. We can filter our `raw_data` RDD as follows.  

# In[2]:

normal_raw_data = raw_data.filter(lambda x: 'normal.' in x)


# Now we can count how many elements we have in the new RDD.

# In[3]:

from time import time
t0 = time()
normal_count = normal_raw_data.count()
tt = time() - t0
print("There are {} 'normal' interactions".format(normal_count))
print("Count completed in {} seconds".format(round(tt,3)))


# Remember from notebook 1 that we have a total of 494021 in our 10 percent dataset. Here we can see that 97278 contain the `normal.` tag word.  

# Notice that we have measured the elapsed time for counting the elements in the RDD. We have done this because we wanted to point out that actual (distributed) computations in Spark take place when we execute *actions* and not *transformations*. In this case `count` is the action we execute on the RDD. We can apply as many transformations as we want on a our RDD and no computation will take place until we call the first action that, in this case takes a few seconds to complete.

# ## The `map` transformation

# By using the `map` transformation in Spark, we can apply a function to every element in our RDD. Python's lambdas are specially expressive for this particular.

# In this case we want to read our data file as a CSV formatted one. We can do this by applying a lambda function to each element in the RDD as follows.

# In[4]:

from pprint import pprint
csv_data = raw_data.map(lambda x: x.split(","))
t0 = time()
head_rows = csv_data.take(5)
tt = time() - t0
print("Parse completed in {} seconds".format(round(tt,3)))
print(head_rows[0])


# Again, all action happens once we call the first Spark *action* (i.e. *take* in this case). What if we take a lot of elements instead of just the first few?  

# In[5]:

t0 = time()
head_rows = csv_data.take(100000)
tt = time() - t0
print("Parse completed in {} seconds".format(round(tt,3)))


# We can see that it takes longer. The `map` function is applied now in a  distributed way to a lot of elements on the RDD, hence the longer execution time.

# ### Using `map` and predefined functions

# Of course we can use predefined functions with `map`. Imagine we want to have each element in the RDD as a key-value pair where the key is the tag (e.g. *normal*) and the value is the whole list of elements that represents the row in the CSV formatted file. We could proceed as follows.    

# In[6]:

def parse_interaction(line):
    elems = line.split(",")
    tag = elems[41]
    return (tag, elems)

key_csv_data = raw_data.map(parse_interaction)
head_rows = key_csv_data.take(5)
print(head_rows[0])


# That was easy, wasn't it?

# In our notebook about working with key-value pairs we will use this type of RDDs to do data aggregations (e.g. count by key).

# ## The `collect` action

# So far we have used the actions `count` and `take`. Another basic action we need to learn is `collect`. Basically it will get all the elements in the RDD into memory for us to work with them. For this reason it has to be used with care, specially when working with large RDDs.  

# An example using our raw data.    

# In[9]:

t0 = time()
all_raw_data = raw_data.collect()
tt = time() - t0
print("Data collected in {} seconds".format(round(tt,3)))


# That took longer as any other action we used before, of course. Every Spark worker node that has a fragment of the RDD has to be coordinated in order to retrieve its part, and then *reduce* everything together.    

# As a last example combining all the previous, we want to collect all the `normal` interactions as key-value pairs.   

# In[13]:

# get data from file
#data_file = "./kddcup.data_10_percent.gz"
data_file = "D:/data_and_dep/kddcup.data_10_percent.gz"
raw_data = sc.textFile(data_file)

# parse into key-value pairs
key_csv_data = raw_data.map(parse_interaction)

# filter normal key interactions
normal_key_interactions = key_csv_data.filter(lambda x: x[0] == "normal.")

# collect all
t0 = time()
all_normal = normal_key_interactions.collect()
tt = time() - t0
normal_count = len(all_normal)
print("Data collected in {} seconds".format(round(tt,3)))
print("There are {} 'normal' interactions".format(normal_count))


# This count matches with the previous count for `normal` interactions. The new procedure is more time consuming. This is because we retrieve all the data with `collect` and then use Python's `len` on the resulting list. Before we were just counting the total number of elements in the RDD by using `count`.  
