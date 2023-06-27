import re
import sys
import pandas as pd
import string
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
from tensorflow.python.keras.layers import LSTM, GRU, Dense, Embedding, Dropout
from tensorflow.python.keras.models import Sequential
from nltk.tokenize import word_tokenize
import numpy as np
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#URLS_RE = re.compile(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b')
URLS_RE = re.compile(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*')

LISTING_RE = re.compile(r'^(|[a-z]?|[0-9]{0,3})(\-|\.)+( |\n)')

def remove_urls(text):
    return URLS_RE.sub('', text)

def replace_multi_whitespaces(line):
    return ' '.join(line.split())

def remove_listing(line):
    return LISTING_RE.sub('', line)

def remove_punctuation(text):
    text = text.replace('!','')
    text = text.replace('"','')
    return text.translate(str.maketrans('','',string.punctuation))

def normalize(s):
    replacements = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
        ("ü","u"),
        ("ñ","n"),
        ("ç","c"),
        ("\u2026","..."),
    )
    for a, b in replacements:
        s = s.replace(a, b).replace(a.upper(), b.upper())
    return s

def remove_stopwords(text,stop_words):
    words = text.split(' ')
    not_stop_words = []
    for word in words:
        if word not in stop_words:
            not_stop_words.append(word)
    return ' '.join(not_stop_words)
      

def clean_text(text,stop_words):	

    low = text.lower()
    norm = normalize(low)
    rem_u = remove_urls(norm)
    rem_l = remove_listing(rem_u)
    rem_w = replace_multi_whitespaces(rem_l)
    rem_p = remove_punctuation(rem_w)
    rem_s = remove_stopwords(rem_p,stop_words)
    text_enc = rem_s.encode('ascii', 'ignore')

    return text_enc.decode()

def tokenized(data):
    tokenized = []
    for text in data:
        tokenized.append(' '.join([token for token in word_tokenize(text,language='spanish',preserve_line=True)]))
    return tokenized

# load spanish stop words and remove accents (tweets dont have accents)
stop_words_df = pd.read_csv('../spanish-stop-words.txt',header=None)
stop_words = [normalize(w) for w in list(stop_words_df[0])] + ['q','ma']

data = pd.read_excel('../cleaned_users.xlsx')
username_list = data['username']
age_list = data['age']

i = 0
age_dict = {}
for age in age_list.unique():
    age_dict[age] = i
    i += 1

all_texts = []
all_age = []
for username,age in zip(username_list,age_list):
    user_age = age_dict.get(age)
    with open(f'../Cleaned Documents/{username}.txt','r') as f:
        for text in f.readlines():
            cleaned = clean_text(text,stop_words)
            all_texts.append(cleaned)
            all_age.append(user_age)

df = pd.DataFrame()
df['text'] = all_texts
df['age'] = all_age

X_t, X_test, y, y_test = train_test_split(df['text'],df['age'],shuffle=True,stratify=df['age'],test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_t,y,shuffle=True,test_size=0.2)

X_train_tokenized = tokenized(X_train)
X_val_tokenized = tokenized(X_val)
X_test_tokenized = tokenized(X_test)
print('tokenized')

max_words = 5000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train_tokenized)
X_train_seq = tokenizer.texts_to_sequences(X_train_tokenized)
X_val_seq = tokenizer.texts_to_sequences(X_val_tokenized)
X_test_seq = tokenizer.texts_to_sequences(X_test_tokenized)
print('sequenced')

max_len = X_train.apply(lambda x: len(x)).max()
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_val_pad = pad_sequences(X_val_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)
word_index = tokenizer.word_index
print('padding')

embedding_vectors = {}
with open('glove-sbwc.i25.vec','r') as f:
    first_line = f.readline().split(' ')
    for line in f.readlines()[1:]:
        row = line.split(' ')
        # remove accents
        word = normalize(row[0])
        weights = np.asarray([float(val) for val in row[1:]])
        embedding_vectors[word] = weights
    
    num_words = int(first_line[0])
    emb_dim = int(first_line[1])

print('loaded GloVe')

if max_words is not None: 
    vocab_len = max_words 
else:
    vocab_len = len(word_index)+1
    
embedding_matrix = np.zeros((vocab_len, emb_dim))
for word, idx in word_index.items():
    if idx < vocab_len:
        embedding_vector = embedding_vectors.get(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector


lstm_model = Sequential()
embedding = Embedding(vocab_len, emb_dim, trainable = False, weights=[embedding_matrix])
embedding.build(vocab_len)
lstm_model.add(embedding)
lstm_model.add(LSTM(128, return_sequences=False))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(1, activation = 'sigmoid'))
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(lstm_model.summary())

print('fitting model...')
epochs  = 5
batch_size = 256
history = lstm_model.fit(X_train_pad, np.asarray(y_train), validation_data=(X_val_pad, np.asarray(y_val)),batch_size=batch_size,epochs = epochs)
print('done')

train_lstm_results = lstm_model.evaluate(X_train_pad, np.asarray(y_train), verbose=0)
val_lstm_results = lstm_model.evaluate(X_val_pad, np.asarray(y_val), verbose=0)
print(f'Train accuracy: {train_lstm_results[1]*100:0.2f}')
print(f'Val accuracy: {val_lstm_results[1]*100:0.2f}')

# predictions
y_pred = lstm_model.predict(X_test_pad)

final_pred = []
for pred in y_pred:
    if pred[0] >= 0.5:
        final_pred.append(1)
    else:
        final_pred.append(0)

acc = accuracy_score(list(y_test),final_pred)
prec = precision_score(list(y_test),final_pred)
rec = recall_score(list(y_test),final_pred)
f1 = f1_score(list(y_test),final_pred)

with open('results.txt','a') as f:
    f.write(f'\t-> Accuracy: {acc}\n')
    f.write(f'\t-> Precision: {prec}\n')
    f.write(f'\t-> Recall: {rec}\n')
    f.write(f'\t-> F1-score: {f1}\n')