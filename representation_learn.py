from pyspark.sql import SparkSession
import pyspark.sql.functions as fn
import boto3 as aws
from processing import create_events_dict, ip2int, macaddr2int
import pandas

import matplotlib.pyplot as plt
plt.style.use('ggplot')

spark = SparkSession\
.builder\
.appName("Representation Learning")\
.enableHiveSupport()\
.config("spark.executor.memory", "2g")\
.config("spark.driver.memory", "2g")\
.config("spark.executor.instances" ,"1")\
.config("spark.executor.cores" , "2")\
.getOrCreate()

# Loading PCAP Data
attack1100 = spark.read.format('csv')\
    .load('s3://prd-power-services-engineering-dsws/212027161/ftp_bgskrot_ex/attack_1100.csv', header='true')

normal1200 = spark.read.format('csv')\
    .load('s3://prd-power-services-engineering-dsws/212027161/ftp_bgskrot_ex/normal_1200.csv', header='true')

pcap_df = attack1100.union(normal1200)
print("Initial PCAP Schema\n")
pcap_df.printSchema()

# Loading Events Data
response = aws.client('s3').get_object(Bucket="prd-power-services-engineering-dsws",
                                 Key="212027161/ftp_bgskrot_ex/exported_events.xml")

data = create_events_dict(response['Body'].read())
events_df = spark.createDataFrame(data)
print("Initial Events Schema\n")
events_df.printSchema()

#Transform Data
pcap_df = pcap_df.withColumnRenamed("frame.time", "datetime")
pcap_df = pcap_df.withColumnRenamed("frame.number", "frame_num")
pcap_df = pcap_df.withColumnRenamed("frame.len", "frame_length")
pcap_df = pcap_df.withColumnRenamed("eth.src", "eth_src")
pcap_df = pcap_df.withColumnRenamed("eth.src", "eth_src")
pcap_df = pcap_df.withColumnRenamed("eth.dst", "eth_dst")
pcap_df = pcap_df.withColumnRenamed("ip.src", "src_ip")
pcap_df = pcap_df.withColumnRenamed("tcp.srcport", "src_port")
pcap_df = pcap_df.withColumnRenamed("ip.dst", "dst_ip")
pcap_df = pcap_df.withColumnRenamed("tcp.dstport", "dst_port")
pcap_df = pcap_df.withColumnRenamed("tcp.window_size", "window_size")
pcap_df = pcap_df.withColumnRenamed("ip.hdr_len", "header_length")
pcap_df = pcap_df.withColumnRenamed("_ws.col.Info", "info")

pd = pcap_df.toPandas()
pd.frame_num = pandas.to_numeric(pd.frame_num, errors='coerce')
pd.frame_length = pandas.to_numeric(pd.frame_length, errors='coerce')
pd.frame_num = pandas.to_numeric(pd.frame_num, errors='coerce')
pd.eth_src = macaddr2int(pd.eth_src)
pd.eth_dst = macaddr2int(pd.eth_dst)
pd.src_ip = ip2int(pd.src_ip)
pd.src_port = pandas.to_numeric(pd.src_port, errors='coerce')
pd.dst_ip = ip2int(pd.dst_ip)
pd.dst_port = pandas.to_numeric(pd.dst_port, errors='coerce')
pd.window_size = pandas.to_numeric(pd.window_size, errors='coerce')
pd.header_length = pandas.to_numeric(pd.header_length, errors='coerce')

print("Updated PCAP Schema\n")
pcap_df = spark.createDataFrame(pd)
pcap_df.printSchema()

ed = events_df.toPandas()
ed.value = pandas.to_numeric(ed.value, errors='coerce')

print("Updated Events Schema\n")
events_df = spark.createDataFrame(ed)
events_df.printSchema()

# Exploring Data
# Show samples of data
pcap_df.show(5)
print("\n=========================================\n")
events_df.show(5)

# Create temporary views for querying
pcap_df.createOrReplaceTempView("pcap")
events_df.createOrReplaceTempView("events")

# Querying data
events_df.select("datetime", "lastvalue").filter(events_df.lastvalue.contains("10.64.88.3")).collect()

# Statistically describe the data
headers = ['frame_length', 'eth_src', 'eth_dst', 'src_ip', 'src_port']
desc = pcap_df.describe(headers)
desc.show()

headers = ['dst_ip', 'dst_port', 'window_size', 'header_length', 'header_length']
desc = pcap_df.describe(headers)
desc.show()

# Correlation Matrix

headers = ['frame_length', 'eth_src', 'eth_dst', 'src_ip', 'src_port', 'dst_ip', 'dst_port',
           'window_size', 'header_length', 'header_length']
num = len(headers)
corr = []

for i in range(0, num):
    temp = [None] * i

    for j in range(i, num):
        temp.append(pcap_df.corr(headers[i], headers[j]))
    corr.append(temp)

print(corr)

# Validate if there are any duplicates

print('Row count: {0}'.format(pcap_df.count()))
print('Distinct rows: {0}'.format(pcap_df.distinct().count()))

pcap_df = pcap_df.dropDuplicates()
print('Row count: {0}'.format(pcap_df.count()))
print('Distinct rows: {0}'.format(pcap_df.distinct().count()))

# Validate there are no duplicate ID's

print('Datetime count: {0}'.format(pcap_df.count()))
print('Distinct Datetime entries {0}'.format(pcap_df.select([c for c in pcap_df.columns if c != 'datetime'])\
                                             .distinct().count()))

pcap_df = pcap_df.dropDuplicates(subset=[c for c in pcap_df.columns if c != 'datetime'])
print('Datetime count: {0}'.format(pcap_df.count()))
print('Distinct Datetime entries {0}'.format(pcap_df.select([c for c in pcap_df.columns if c != 'datetime'])\
                                             .distinct().count()))

pcap_df.agg(
    fn.count('datetime').alias('count'),
    fn.countDistinct('datetime').alias('distinct')
).show()

#Visualizations
hist = pcap_df.select('window_size').rdd.flatMap(lambda row : row).histogram(20)
data = {'bins': hist[0][: -1], 'freq': hist[1]}
plt.bar(data['bins'], data['freq'], width=2000)
plt.title('Matplotlib Histogram of \'window_size\'')

# Preparing data for model
pcap_pandas = pcap_df.toPandas()
pcap_pandas= pcap_pandas.fillna(method='backfill')
pcap_X = pcap_pandas.iloc[:,1:10]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
pcap_X = sc.fit_transform(pcap_X)
pcap_X.shape

# lstm autoencoder recreate sequence
from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model

# define input sequence
sequence = pcap_X

# reshape input into [samples, timesteps, features]
n_in = 8
sequence = sequence.reshape((384757, n_in, 9))

# define model
model = Sequential()
model.add(LSTM(8, activation='relu', input_shape=(n_in,9)))
model.add(RepeatVector(n_in))
model.add(LSTM(8, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(9)))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(sequence, sequence, epochs=300, verbose=0)
# connect the encoder LSTM as the output layer
model = Model(inputs=model.inputs, outputs=model.layers[0].output)
# plot_model(model, show_shapes=True, to_file='lstm_encoder.png')
# get the feature vector for the input sequence
yhat = model.predict(sequence)
print(yhat.shape)
print(yhat)

spark.stop()