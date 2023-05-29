import numpy as np
import pandas as pd
import src.processing as proc

from pyspark import SparkConf, SparkContext, SparkSession
from elephas.utils.rdd_utils import to_simple_rdd
from elephas.spark_model import SparkModel
from keras.models import Sequential
from keras.layers import LSTM, Dense, Softmax
from keras.utils import np_utils as kutils
from matplotlib import pyplot as plt

# Data Formatting
# pcap data
SparkSession.
pcap_labels = ['datetime', 'highest_layer', 'transport_layer', 'src_ip', 'src_port', 'dst_ip',
               'dst_port', 'packet_length', 'window_size', 'header_length']
pcap_df = proc.pcap_data('/Users/patrickday/Research/data-analysis/src/resources/pcap/SMIA_2012_1127_1143.pcap', pcap_labels)
pcap_df = pcap_df.set_index('datetime')
pcap_df.highest_layer = pd.Categorical(pcap_df.highest_layer).codes
pcap_df.transport_layer = pd.Categorical(pcap_df.transport_layer).codes

# netflow log data
netflow_labels = ['datetime', 'duration', 'protocol', 'src_ip', 'src_port', 'dst_ip',
                  'dst_port', 'packets', 'bytes', 'flows']
netflow_df = proc.netflow_log_data(
    '/Users/patrickday/Research/data-analysis/src/resources/logs/netflow_172.30.13.10.log', netflow_labels)
netflow_df = netflow_df.set_index('datetime')
netflow_df.protocol = pd.Categorical(netflow_df.protocol).codes

# Train & Test split
pcap_msk = np.random.rand(len(pcap_df)) < 0.8
pcap_x_train = pcap_df[pcap_msk]
pcap_x_test = pcap_df[~pcap_msk]

netflow_msk = np.random.rand(len(netflow_df)) < 0.8
netflow_x_train = netflow_df[netflow_msk]
netflow_x_test = netflow_df[~netflow_msk]

# Create a Spark ML Context
conf = SparkConf().setAppName('Python ML').setMaster('local[8]')
spark_context = SparkContext(conf=conf)

# Model Architecture
# pcap data set
pcap_model = Sequential()
pcap_model.add(LSTM(100, return_sequences=True))
pcap_model.add(Dense)
pcap_model.add(Softmax)
pcap_model.compile(optimizer='adam', loss='mean_squared_error')

pcap_rdd = to_simple_rdd(spark_context, pcap_x_train, pcap_labels)
model = SparkModel(pcap_model, frequency='epoch', mode='asynchronous')
model.fit(pcap_rdd, epochs=20, batch_size=32, verbose=0, validation_split=0.1)

# Learned layers to transfer
transferredLayers = pcap_model.layers.__getitem__(0)
transferredLayers.trainable = False

# log data set
netflow_model = Sequential()
netflow_model.add(transferredLayers)
netflow_model.add(LSTM(100, return_sequences=True))
netflow_model.add(Dense)
netflow_model.add(Softmax)
netflow_model.compile(optimizer='adam', loss='mean_squared_error')

netflow_rdd = to_simple_rdd(spark_context, netflow_x_train, netflow_labels)
model = SparkModel(netflow_model, frequency='epoch', mode='asynchronous')
model.fit(netflow_rdd, epochs=20, batch_size=32, verbose=0, validation_split=0.1)
