
# coding: utf-8

# # MLlib: Basic Statistics and Exploratory Data Analysis

# [Introduction to Spark with Python, by Jose A. Dianes](https://github.com/jadianes/spark-py-notebooks)

# So far we have used different map and aggregation functions, on simple and key/value pair RDD's, in order to get simple statistics that help us understand our datasets. In this notebook we will introduce Spark's machine learning library [MLlib](https://spark.apache.org/docs/latest/mllib-guide.html) through its basic statistics functionality in order to better understand our dataset. We will use the reduced 10-percent [KDD Cup 1999](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) datasets through the notebook.   

# ## Getting the data and creating the RDD

# As we did in our first notebook, we will use the reduced dataset (10 percent) provided for the [KDD Cup 1999](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html), containing nearly half million network interactions. The file is provided as a Gzip file that we will download locally.  

# In[1]:
from pyspark import SparkContext
from pyspark import SparkConf
print ("Successfully imported Spark Modules")

from operator import add

conf = SparkConf().setMaster("local").setAppName("My App").set("spark.executor.extraJavaOptions", "-Dfile.encoding=UTF-8")
sc=SparkContext("local")

import urllib
#f = urllib.urlretrieve ("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz", "kddcup.data_10_percent.gz")


# In[2]:

#data_file = "./kddcup.data_10_percent.gz"
data_file = "D:/data_and_dep/kddcup.data_10_percent.gz"
raw_data = sc.textFile(data_file)


# ## Local vectors

# A [local vector](https://spark.apache.org/docs/latest/mllib-data-types.html#local-vector) is often used as a base type for RDDs in Spark MLlib. A local vector has integer-typed and 0-based indices and double-typed values, stored on a single machine. MLlib supports two types of local vectors: dense and sparse. A dense vector is backed by a double array representing its entry values, while a sparse vector is backed by two parallel arrays: indices and values. 

# For dense vectors, MLlib uses either Python *lists* or the *NumPy* `array` type. The later is recommended, so you can simply pass NumPy arrays around.  

# For sparse vectors, users can construct a `SparseVector` object from MLlib or pass *SciPy* `scipy.sparse` column vectors if SciPy is available in their environment. The easiest way to create sparse vectors is to use the factory methods implemented in `Vectors`.  

# ### An RDD of dense vectors

# Let's represent each network interaction in our dataset as a dense vector. For that we will use the *NumPy* `array` type.  

# In[3]:

import numpy as np

def parse_interaction(line):
    line_split = line.split(",")
    # keep just numeric and logical values
    symbolic_indexes = [1,2,3,41]
    clean_line_split = [item for i,item in enumerate(line_split) if i not in symbolic_indexes]
    return np.array([float(x) for x in clean_line_split])

vector_data = raw_data.map(parse_interaction)


# ## Summary statistics

# Spark's MLlib provides column summary statistics for `RDD[Vector]` through the function [`colStats`](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.stat.Statistics.colStats) available in [`Statistics`](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.stat.Statistics). The method returns an instance of [`MultivariateStatisticalSummary`](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.stat.MultivariateStatisticalSummary), which contains the column-wise *max*, *min*, *mean*, *variance*, and *number of nonzeros*, as well as the *total count*.  

# In[4]:

from pyspark.mllib.stat import Statistics 
from math import sqrt 

# Compute column summary statistics.
summary = Statistics.colStats(vector_data)

print("Duration Statistics:")
print(" Mean: {}".format(round(summary.mean()[0],3)))
print(" St. deviation: {}".format(round(sqrt(summary.variance()[0]),3)))
print(" Max value: {}".format(round(summary.max()[0],3)))
print(" Min value: {}".format(round(summary.min()[0],3)))
print(" Total value count: {}".format(summary.count()))
print(" Number of non-zero values: {}".format(summary.numNonzeros()[0]))


# ### Summary statistics by label  

# The interesting part of summary statistics, in our case, comes from being able to obtain them by the type of network attack or 'label' in our dataset. By doing so we will be able to better characterise our dataset dependent variable in terms of the independent variables range of values.  

# If we want to do such a thing we could filter our RDD containing labels as keys and vectors as values. For that we just need to adapt our `parse_interaction` function to return a tuple with both elements.     

# In[5]:

def parse_interaction_with_key(line):
    line_split = line.split(",")
    # keep just numeric and logical values
    symbolic_indexes = [1,2,3,41]
    clean_line_split = [item for i,item in enumerate(line_split) if i not in symbolic_indexes]
    return (line_split[41], np.array([float(x) for x in clean_line_split]))

label_vector_data = raw_data.map(parse_interaction_with_key)


# The next step is not very sophisticated. We use `filter` on the RDD to leave out other labels but the one we want to gather statistics from.    

# In[6]:

normal_label_data = label_vector_data.filter(lambda x: x[0]=="normal.")


# Now we can use the new RDD to call `colStats` on the values.  

# In[7]:

normal_summary = Statistics.colStats(normal_label_data.values())


# And collect the results as we did before.  

# In[8]:

print("Duration Statistics for label: {}".format("normal"))
print(" Mean: {}".format(normal_summary.mean()[0],3))
print(" St. deviation: {}".format(round(sqrt(normal_summary.variance()[0]),3)))
print(" Max value: {}".format(round(normal_summary.max()[0],3)))
print(" Min value: {}".format(round(normal_summary.min()[0],3)))
print(" Total value count: {}".format(normal_summary.count()))
print(" Number of non-zero values: {}".format(normal_summary.numNonzeros()[0]))


# Instead of working with a key/value pair we could have just filter our raw data split using the label in column 41. Then we can parse the results as we did before. This will work as well. However having our data organised as key/value pairs will open the door to better manipulations. Since `values()` is a transformation on an RDD, and not an action, we don't perform any computation until we call `colStats` anyway.  

# But lets wrap this within a function so we can reuse it with any label.

# In[9]:

def summary_by_label(raw_data, label):
    label_vector_data = raw_data.map(parse_interaction_with_key).filter(lambda x: x[0]==label)
    return Statistics.colStats(label_vector_data.values())


# Let's give it a try with the "normal." label again.  

# In[10]:

normal_sum = summary_by_label(raw_data, "normal.")

print("Duration Statistics for label: {}".format("normal"))
print(" Mean: {}".format(normal_sum.mean()[0],3))
print(" St. deviation: {}".format(round(sqrt(normal_sum.variance()[0]),3)))
print(" Max value: {}".format(round(normal_sum.max()[0],3)))
print(" Min value: {}".format(round(normal_sum.min()[0],3)))
print(" Total value count: {}".format(normal_sum.count()))
print(" Number of non-zero values: {}".format(normal_sum.numNonzeros()[0]))


# Let's try now with some network attack. We have all of them listed [here](http://kdd.ics.uci.edu/databases/kddcup99/training_attack_types).  

# In[11]:

guess_passwd_summary = summary_by_label(raw_data, "guess_passwd.")

print("Duration Statistics for label: {}".format("guess_password"))
print(" Mean: {}".format(guess_passwd_summary.mean()[0],3))
print(" St. deviation: {}".format(round(sqrt(guess_passwd_summary.variance()[0]),3)))
print(" Max value: {}".format(round(guess_passwd_summary.max()[0],3)))
print(" Min value: {}".format(round(guess_passwd_summary.min()[0],3)))
print(" Total value count: {}".format(guess_passwd_summary.count()))
print(" Number of non-zero values: {}".format(guess_passwd_summary.numNonzeros()[0]))


# We can see that this type of attack is shorter in duration than a normal interaction. We could build a table with duration statistics for each type of interaction in our dataset. First we need to get a list of labels as described in the first line [here](http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names).      

# In[12]:

label_list = ["back.","buffer_overflow.","ftp_write.","guess_passwd.",
              "imap.","ipsweep.","land.","loadmodule.","multihop.",
              "neptune.","nmap.","normal.","perl.","phf.","pod.","portsweep.",
              "rootkit.","satan.","smurf.","spy.","teardrop.","warezclient.",
              "warezmaster."]


# Then we get a list of statistics for each label.  

# In[13]:

stats_by_label = [(label, summary_by_label(raw_data, label)) for label in label_list]


# Now we get the *duration* column, first in our dataset (i.e. index 0).  

# In[14]:

duration_by_label = [ 
    (stat[0], np.array([float(stat[1].mean()[0]), float(sqrt(stat[1].variance()[0])), float(stat[1].min()[0]), float(stat[1].max()[0]), int(stat[1].count())])) 
    for stat in stats_by_label]


# That we can put into a Pandas data frame.  

# In[15]:

import pandas as pd
pd.set_option('display.max_columns', 50)

stats_by_label_df = pd.DataFrame.from_items(duration_by_label, columns=["Mean", "Std Dev", "Min", "Max", "Count"], orient='index')


# And print it.

# In[16]:

print("Duration statistics, by label")
stats_by_label_df


# In order to reuse this code and get a dataframe from any variable in our dataset we will define a function.  

# In[17]:

def get_variable_stats_df(stats_by_label, column_i):
    column_stats_by_label = [
        (stat[0], np.array([float(stat[1].mean()[column_i]), float(sqrt(stat[1].variance()[column_i])), float(stat[1].min()[column_i]), float(stat[1].max()[column_i]), int(stat[1].count())])) 
        for stat in stats_by_label
    ]
    return pd.DataFrame.from_items(column_stats_by_label, columns=["Mean", "Std Dev", "Min", "Max", "Count"], orient='index')


# Let's try for *duration* again.   

# In[18]:

get_variable_stats_df(stats_by_label,0)


# Now for the next numeric column in the dataset, *src_bytes*.  

# In[19]:

print("src_bytes statistics, by label")
get_variable_stats_df(stats_by_label,1)
