# -*- coding:utf-8 -*- 
import os
import sys
# Path for spark source folder
os.environ['SPARK_HOME']="D:\data_and_dep\sparkml\software\spark-2.1.0-bin-hadoop2.7\spark-2.1.0-bin-hadoop2.7"
os.environ['HADOOP_HOME']="D:\data_and_dep\sparkml\software\spark-2.1.0-bin-hadoop2.7\spark-2.1.0-bin-hadoop2.7"

# # Append pyspark  to Python Path
sys.path.append("D:\data_and_dep\sparkml\software\spark-2.1.0-bin-hadoop2.7\spark-2.1.0-bin-hadoop2.7\python")
sys.path.append("D:\data_and_dep\sparkml\software\spark-2.1.0-bin-hadoop2.7\spark-2.1.0-bin-hadoop2.7\python\lib/py4j-0.9-src.zip")


from pyspark import SparkContext
from pyspark import SparkConf
print ("Successfully imported Spark Modules")

from operator import add

#conf = SparkConf().setMaster("local").setAppName("My App").set("spark.executor.extraJavaOptions", "-Dfile.encoding=UTF-8")
sc=SparkContext("local")
lines=sc.textFile('D:/project/peixun/ai_course_project_px/8_sparkml/0_classification/README.md')
tmp=lines.flatMap(lambda x:x.split(' ')).map(lambda x:(x,1))

counts=tmp.reduceByKey(add)

output=counts.collect();
for (word,count) in output:
    print("xxx: %s %i" % (word,count))
sc.stop()