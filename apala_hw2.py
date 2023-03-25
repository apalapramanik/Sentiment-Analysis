# -*- coding: utf-8 -*-
"""apala_hw2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kkhWueZO7bSOEifm8Up1n_x0FK7EG-KX
"""

import os
import matplotlib.pyplot as plt  
import numpy as np 
import tensorflow as tf  
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from keras.datasets import imdb
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import regularizers
from keras.utils.vis_utils import plot_model
from keras.layers import *
from keras.models import *
from keras import backend as K
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential



os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

n_unique_words = 10000
(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=n_unique_words)

maxlen = 200
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
y_train = np.array(y_train)
y_test = np.array(y_test)

class attention(Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences

        super(attention,self).__init__()

    def build(self, input_shape):
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),initializer="normal")
        self.b= self.add_weight(name="att_bias", shape=(input_shape[1],1),initializer="normal")
        self.b= self.add_weight(name="att_bias", shape=(input_shape[1],1))        
        super(attention,self).build(input_shape)


    def call(self, x):
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        if self.return_sequences:
            return output
        return K.sum(output, axis=1)

model = Sequential()
model.add(Embedding(n_unique_words, 128, input_length=maxlen))
model.add(Bidirectional(GRU(128, return_sequences=True)))
model.add(attention(return_sequences=False)) # receive 3D and output 3D
model.add(Dropout(0.5))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
model.summary()

history=model.fit(x_train, y_train,batch_size=32,epochs=10)
print(history.history['loss'])
print(history.history['accuracy']) 
# model.save(f"/work/cse479/apramani/model_gru_4.h5")

# test data
loss, accuracy = model.evaluate(x_test, y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

Y_pred = model.predict(x_test)
Y_pred = Y_pred.ravel() 
y_pred_binary = np.where(Y_pred >= 0.5, 1, 0)

import seaborn as sns
conf_matrix = confusion_matrix(y_test, y_pred_binary)
tn, fp, fn, tp = conf_matrix.ravel()

sns.set(font_scale=1.4)
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='BuPu', 
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

