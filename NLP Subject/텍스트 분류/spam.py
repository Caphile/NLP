
# -*- coding: cp949 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

urllib.request.urlretrieve("https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv", filename="spam.csv")

data = pd.read_csv('spam.csv',encoding='latin1')

print('�� ������ �� :',len(data))

print(data[:5])
data['v1'] = data['v1'].replace(['ham','spam'],[0,1])

X_data = data['v2']
y_data = data['v1']

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_data) # X�� �� �࿡ ��ūȭ�� ����
sequences = tokenizer.texts_to_sequences(X_data) # �ܾ ���ڰ�, �ε����� ��ȯ�Ͽ� ����

word_to_index = tokenizer.word_index

vocab_size = len(word_to_index) + 1

n_of_train = int(len(sequences) * 0.8)
n_of_test = int(len(sequences) - n_of_train)

X_data = sequences

max_len = max(len(l) for l in X_data)
# ��ü �����ͼ��� ���̴� max_len���� ����ϴ�.
data = pad_sequences(X_data, maxlen = max_len)
print("�Ʒ� �������� ũ��(shape): ", data.shape)

X_test = data[n_of_train:] #X_data ������ �߿��� ���� 1034���� �����͸� ����
y_test = np.array(y_data[n_of_train:]) #y_data ������ �߿��� ���� 1034���� �����͸� ����
X_train = data[:n_of_train] #X_data ������ �߿��� ���� 4135���� �����͸� ����
y_train = np.array(y_data[:n_of_train]) #y_data ������ �߿��� ���� 4135���� �����͸� ����

from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Embedding(vocab_size, 32)) # �Ӻ��� ������ ������ 32
model.add(SimpleRNN(32)) # RNN ���� hidden_size�� 32
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=4, batch_size=64, validation_split=0.2)
