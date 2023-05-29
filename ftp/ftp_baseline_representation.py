from probe.utils import split_train_test, scale_df
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

import matplotlib.pyplot as plt
import pandas

#Load & Prepare Data
labeled = pandas.read_csv("/Users/patrickday/Research/autonomic-python-probe/resources/ftp_bgskrot_ex/datasets/labeled.csv")
unlabeled = pandas.read_csv("/Users/patrickday/Research/autonomic-python-probe/resources/ftp_bgskrot_ex/datasets/unlabeled.csv")
training, test = split_train_test(labeled, 0.2)

training_labels = training['attack']
training = scale_df(training.loc[:, "frame_num" : "info"])
training['attack'] = training_labels
training = training.dropna(subset=["attack"])

test_labels = test['attack']
test = scale_df(test.loc[:, "frame_num" : "info"])
test['attack'] = test_labels
test = training.dropna(subset=['attack'])

# define input sequence
unlabeled = unlabeled.iloc[:460000,2:13]
print(unlabeled.shape)
ulabeled = unlabeled.values
print(ulabeled)

# reshape input into [samples, timesteps, features]
n_in = 32
ulabeled = ulabeled.reshape((14375, n_in, 11))

# define model
# Encoding Section
pcap_model = Sequential()
pcap_model.add(LSTM(11, activation='relu', input_shape=(n_in,11), name='ulabeled_pcap'))

# Decoding Section
pcap_model.add(RepeatVector(n_in))
pcap_model.add(LSTM(11, activation='relu', return_sequences=True, name='decode'))
pcap_model.add(TimeDistributed(Dense(11)))
pcap_model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, clipnorm=1.0), loss='mse')
pcap_model.summary()

# fit model
performance = pcap_model.fit(ulabeled, ulabeled, epochs=50, verbose=1)
pcap_model.save("/Users/patrickday/Research/autonomic-python-probe/resources/models/pcap_model.h5")

# Plot training & validation accuracy values
# plt.plot(performance.history['acc'])
# plt.plot(performance.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# Plot training & validation loss values
plt.plot(performance.history['loss'])
# plt.plot(performance.history['val_loss'])
plt.title('UNSW-B15 Baseline Representation Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.grid(True)
plt.show()

# connect the encoder LSTM as the output layer
# pcap_final_model = Model(inputs=pcap_model.inputs, outputs=pcap_model.layers[0].output)
# pcap_final_model.save("/Users/patrickday/Research/autonomic-python-probe/resources/models/pcap_final_model.h5")

# get the feature vector for the input sequence
# yhat = pcap_final_model.predict(ulabeled)
# print(yhat.shape)
# print(yhat)