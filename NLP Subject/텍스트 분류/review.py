
# -*- coding: cp949 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
#import urllib.request
from konlpy.tag import Okt
#from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
#urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')

train_data.drop_duplicates(subset=['document'], inplace=True) # document ������ �ߺ��� ������ �ִٸ� �ߺ� ����

print('�� ������ �� :',len(train_data))
print(train_data.groupby('label').size().reset_index(name = 'count'))

train_data = train_data.dropna(how = 'any') # Null ���� �����ϴ� �� ����
print(train_data.isnull().values.any()) # Null ���� �����ϴ��� Ȯ��

train_data['document'] = train_data['document'].str.replace("[^��-����-�Ӱ�-�R ]","")
# �ѱ۰� ������ �����ϰ� ��� ����
train_data[:5]

train_data = train_data.dropna(how = 'any')
print(len(train_data))

test_data.drop_duplicates(subset = ['document'], inplace=True) # document �ߺ� ����
test_data['document'] = test_data['document'].str.replace("[^��-����-�Ӱ�-�R ]","") # ���� ǥ���� ����
test_data['document'].replace('', np.nan, inplace=True) # ������ Null ������ ����
test_data = test_data.dropna(how='any') # Null �� ����
print('��ó�� �� �׽�Ʈ�� ������ ���� :',len(test_data))

stopwords = ['��','��','��','��','��','��','��','��','��','��','��','��','����','��','��','��','��','�ϴ�']

okt = Okt()	# ���¼� �м���

X_train = []
k = 0
for sentence in train_data['document']:
    k = k+1
    if k % 5000 == 0: print(k)
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True) # ��ūȭ
    temp_X = [word for word in temp_X if not word in stopwords] # �ҿ�� ����
    X_train.append(temp_X)  

X_test = []
k = 0
for sentence in test_data['document']:
    k = k+1
    if k % 5000 == 0: print(k)
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True) # ��ūȭ
    temp_X = [word for word in temp_X if not word in stopwords] # �ҿ�� ����
    X_test.append(temp_X)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

threshold = 3
total_cnt = len(tokenizer.word_index) # �ܾ��� ��
rare_cnt = 0 # ���� �󵵼��� threshold���� ���� �ܾ��� ������ ī��Ʈ
total_freq = 0 # �Ʒ� �������� ��ü �ܾ� �󵵼� �� ��
rare_freq = 0 # ���� �󵵼��� threshold���� ���� �ܾ��� ���� �󵵼��� �� ��

# �ܾ�� �󵵼��� ��(pair)�� key�� value�� �޴´�.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # �ܾ��� ���� �󵵼��� threshold���� ������
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

vocab_size = total_cnt - rare_cnt + 1

tokenizer = Tokenizer(vocab_size, oov_token = 'OOV') 
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])

drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]

X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)
print(len(X_train))
print(len(y_train))

max_len = 30
X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)

'''

from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

embedding_dim = 100
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(LSTM(hidden_units))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.2)

loaded_model = load_model('best_model.h5')
print("\n �׽�Ʈ ��Ȯ��: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

'''

from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Embedding(vocab_size, 32)) # �Ӻ��� ������ ������ 32
model.add(SimpleRNN(32)) # RNN ���� hidden_size�� 32
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, batch_size=64, validation_split=0.2)

model.summary()


loaded_model = load_model('best_model.h5')
print("\n �׽�Ʈ ��Ȯ��: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

