
# coding: utf-8

# # Data aggregations on RDDs

# [Introduction to Spark with Python, by Jose A. Dianes](https://github.com/jadianes/spark-py-notebooks)

# We can aggregate RDD data in Spark by using three different actions: `reduce`, `fold`, and `aggregate`. The last one is the more general one and someway includes the first two.  

# ## Getting the data and creating the RDD

# As we did in our first notebook, we will use the reduced dataset (10 percent) provided for the [KDD Cup 1999](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html), containing nearly half million nework interactions. The file is provided as a Gzip file that we will download locally.  
from pyspark import SparkContext
from pyspark import SparkConf
print ("Successfully imported Spark Modules")

from operator import add

conf = SparkConf().setMaster("local").setAppName("My App").set("spark.executor.extraJavaOptions", "-Dfile.encoding=UTF-8")
sc=SparkContext("local")
# In[1]:

import urllib
#f = urllib.urlretrieve ("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz", "kddcup.data_10_percent.gz")


# In[2]:

#data_file = "./kddcup.data_10_percent.gz"
data_file = "D:/data_and_dep/kddcup.data_10_percent.gz"

raw_data = sc.textFile(data_file)


# ## Inspecting interaction duration by tag

# Both `fold` and `reduce` take a function as an argument that is applied to two elements of the RDD. The `fold` action differs from `reduce` in that it gets and additional initial *zero value* to be used for the initial call. This value should be the identity element for the function provided.  

# As an example, imagine we want to know the total duration of our interactions for normal and attack interactions. We can use `reduce` as follows.    

# In[3]:

# parse data
csv_data = raw_data.map(lambda x: x.split(","))

# separate into different RDDs
normal_csv_data = csv_data.filter(lambda x: x[41]=="normal.")
attack_csv_data = csv_data.filter(lambda x: x[41]!="normal.")


# The function that we pass to `reduce` gets and returns elements of the same type of the RDD. If we want to sum durations we need to extract that element into a new RDD.  

# In[4]:

normal_duration_data = normal_csv_data.map(lambda x: int(x[0]))
attack_duration_data = attack_csv_data.map(lambda x: int(x[0]))


# Now we can reduce these new RDDs.  

# In[5]:

total_normal_duration = normal_duration_data.reduce(lambda x, y: x + y)
total_attack_duration = attack_duration_data.reduce(lambda x, y: x + y)

print("Total duration for 'normal' interactions is {}".    format(total_normal_duration))
print("Total duration for 'attack' interactions is {}".    format(total_attack_duration))


# We can go further and use counts to calculate duration means.  

# In[6]:

normal_count = normal_duration_data.count()
attack_count = attack_duration_data.count()

print("Mean duration for 'normal' interactions is {}".    format(round(total_normal_duration/float(normal_count),3)))
print("Mean duration for 'attack' interactions is {}".    format(round(total_attack_duration/float(attack_count),3)))

