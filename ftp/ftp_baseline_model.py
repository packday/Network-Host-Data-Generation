from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

import matplotlib.pyplot as plt
import pandas


#Load Data
data = pandas.read_csv("/Users/patrickday/Research/autonomic-python-probe/resources/ftp_bgskrot_ex/datasets/labeled.csv")
data = data.iloc[:,2:14]

#Load Model
pcap_model = load_model("/Users/patrickday/Research/autonomic-python-probe/resources/models/pcap_model.h5")
transfer = pcap_model.layers[0]
transfer.trainable=False


#Semi-Supervised Model
semi_supervised_model = Sequential()
# semi_supervised_model.add(transfer)
semi_supervised_model.add(LSTM(16, activation='relu', return_sequences=True, name='decode'))
semi_supervised_model.add(Dropout(0.5))
semi_supervised_model.add(Dense(10, activation='relu'))
semi_supervised_model.add(Dense(32, activation='softmax'))
semi_supervised_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, clipnorm=0.5), metrics=['accuracy'])
semi_supervised_model.summary()

#Fit Model
train_df = pandas.read_csv("/Users/patrickday/Research/autonomic-python-probe/resources/UNSW-NB15/UNSW_NB15_training-set.csv")
train_df
x_train = train_df.iloc[:,:48]
x_train = x_train.drop([], axis=1)
# x_train = x_train.values
# x_train = x_train.reshape((99, 32, 11))

y_train = train_df['Label']
y_train = y_train.values
y_train = y_train.reshape((99, 32))


performance = semi_supervised_model.fit(x_train, y_train, epochs=5000, verbose=1)
semi_supervised_model.save("/Users/patrickday/Research/autonomic-python-probe/resources/models/semi_supervised_model.h5")

# Plot training & validation accuracy values
plt.plot(performance.history['acc'])
# plt.plot(performance.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(performance.history['loss'])
# plt.plot(performance.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
