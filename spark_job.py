from pyspark.sql import SparkSession
import matplotlib
 # %matplotlib inline

print("Running Spark Job")
spark = SparkSession\
        .builder\
        .appName("Spark Job")\
        .master("local[2]")\
        .getOrCreate()

df = spark.read.format('csv').load('/Users/patrickday/Research/data-analysis/src/resources/pcap/pcap_100000.csv', inferSchema='true', header='true').cache()
df.show()
df.toPandas().plot.line('datetime', 'packet_length')

# Stop spark session
spark.stop()
print("Completed Spark Job")