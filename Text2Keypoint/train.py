from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Input, Dense,Embedding, Concatenate, TimeDistributed
from tensorflow.keras.models import Model,load_model, model_from_json
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from matplotlib import pyplot
import string
from string import digits
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import json
import re
import os

from tacotron import *

# Load morpheme json =========================================

def preprocess_mor(f):
  f = str(f)
  #print(f)
  f = re.sub(r'[^\w]','',f)
  f = re.sub(r'name',' ',f)
  f = re.sub(r'[a-zA-Z0-9_]','',f)
  f = re.sub(r'  ','',f)
  return f


path_sen_mor = 'path/SEN/json(형태소)'
path_word_mor= 'path/WORD/json(형태소)'

list_sen_mor=os.listdir(path_sen_mor)
list_word_mor=os.listdir(path_word_mor)

mor = []

for l_sen_mor in list_sen_mor:
  with open(path_sen_mor+'/'+l_sen_mor, "r", encoding="utf-8", errors='ignore') as f:
      f = json.load(f) 
      f = preprocess_mor(f)
      mor.append(f)

for l_word_mor in list_word_mor:
  with open(path_word_mor+'/'+l_word_mor, "r", encoding="utf-8", errors='ignore') as f:
      f = json.load(f) 
      f = preprocess_mor(f)
      mor.append(f)

X_data = pd.DataFrame(mor)
X = X_data[0].values

# Load keypoint json =========================================

path_sen_key= 'path/SEN/txt(키포인트)'
path_word_key= 'path/WORD/txt(키포인트)'

list_sen_key=os.listdir(path_sen_key)
list_word_key=os.listdir(path_word_key)

key = []

for l_sen_key in list_sen_key:
  f = open(path_sen_key+'/'+l_sen_key, 'r')
  lines = f.readlines()
  f.close()
  key.append(lines)

for l_word_key in list_word_key:
  f = open(path_word_key+'/'+l_word_key, 'r')
  lines = f.readlines()
  f.close()
  key.append(lines)

y_data = pd.DataFrame(key)
y_data = y_data.transpose()

y=[]

for i in range(20):
    frame= y_data[i].tolist()
    tmp=[]
  
    for j in range(45):
        tmp_list=[]
        
        if frame[j] == None:
            tmp_list=[0 for _ in range(254)]

        else:
            key_frame = frame[j].split(",")
            del key_frame[0]
            for k in range(254):
                tmp_list.append(float(key_frame[k]))

        tmp.append(tmp_list)
    y.append(tmp)
y=np.array(y)

# Data split =========================================

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.1)

def Max_length(data):
  max_length_ = max([len(x.split(' ')) for x in data])
  return max_length_

#Training data
max_X= Max_length(X_train) 
max_y= 254

#Test data
max_X_test = Max_length(X_test)
max_y_test=254 

# Embedding =========================================

Tok = Tokenizer()
Tok.fit_on_texts(X)

word2index = Tok.word_index
vocab_size_source = len(word2index) + 1
vocab_size_target = 254 # num of keypoint in 1 frame

X_train = Tok.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen=max_X, padding='post')
X_test = Tok.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=max_X, padding='post')
X_ = Tok.texts_to_sequences(X)
X_ = pad_sequences(X_, maxlen=max_X, padding='post')

# Model =========================================

N_MEL = max_y
REF_DB = 2 
MAX_DB = 10 
r = 5 
MAX_MEL_TIME_LENGTH = 45 # max num of frames
WINDOW_TYPE='hann' 
N_ITER = 5 

# for Text
NB_CHARS_MAX = max_X

# for Model
K1 = 16 
K2 = 8  
BATCH_SIZE = 1 
NB_EPOCHS = 15 
EMBEDDING_SIZE = 8 
TRAIN_SET_RATIO = 0.9 

latent_dim = 50

encoder = Encoder(max_X, latent_dim, vocab_size_source, K1)
input_encoder, cbhg_encoding = encoder(0, 0)

decoder_prenet = Decoder_prenet(N_MEL, latent_dim, vocab_size_target)
input_decoder, attention_rnn_output = decoder_prenet(0, 0)

attention = Attention(max_X, cbhg_encoding, attention_rnn_output)
attention_context, attention_rnn_output_reshaped = attention(0, 0)

decoder = Decoder(attention_context,
                  attention_rnn_output_reshaped, MAX_MEL_TIME_LENGTH, N_MEL)
mel_hat_ = decoder(0) 

model = Model([input_encoder, input_decoder], outputs=mel_hat_)
opt = Adam()
model.compile(optimizer=opt,
              loss=['mean_absolute_error', 'mean_absolute_error'])



decoder_input = []
mel_spectro_data = []

decod_inp = tf.concat((tf.zeros_like(y[:1, :]),
                       y[:-1, :]), 0)
decod_inp = decod_inp[:, -N_MEL:]

# Padding of the temporal dimension
dim0_mel_spectro = y.shape[0]
dim1_mel_spectro = y.shape[1]
dim2_mel_spectro = y.shape[2]
padded_mel_spectro = np.zeros(
    (dim0_mel_spectro, dim1_mel_spectro, dim2_mel_spectro))
padded_mel_spectro[:dim0_mel_spectro, :dim1_mel_spectro] = y

dim0_decod_inp = decod_inp.shape[0]
dim1_decod_inp = decod_inp.shape[1]
dim2_decod_inp = decod_inp.shape[2]
padded_decod_input = np.zeros((dim0_decod_inp, dim1_decod_inp, dim2_decod_inp))
padded_decod_input[:dim0_decod_inp, :dim1_decod_inp] = decod_inp

decoder_input_array = np.array(padded_decod_input)
mel_spectro_data_array = np.array(padded_mel_spectro)


len_train = int(TRAIN_SET_RATIO * len(y))
print(len_train)
print(decoder_input_array.shape)

decoder_input_array_training = decoder_input_array[:len_train]
decoder_input_array_testing = decoder_input_array[len_train:]

mel_spectro_data_array_training = mel_spectro_data_array[:len_train]
mel_spectro_data_array_testing = mel_spectro_data_array[len_train:]

# Train =========================================

train_history = model.fit([X_, decoder_input_array],
                          mel_spectro_data_array,
                          epochs=3000, batch_size=BATCH_SIZE,
                          verbose=1, validation_data=([X_test, decoder_input_array_testing], mel_spectro_data_array_testing))

pyplot.plot(train_history.history['loss'], label='train')
pyplot.plot(train_history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

model.save('model.h5')

# Predict =========================================

predictions = model.predict([X_, decoder_input_array])

index2word=Tok.index_word

def seq2text(input_seq):
    newString=''
    for i in input_seq:
      if(i!=0):
        newString=newString+index2word[i]+' '
    return newString


for i in range(len(X_)):

    print("num of test data is", i)
    print(seq2text(X_[i]))
    print("test: ", y[i])
    print("predict: ", predictions[i])
    print("===================================")

for i in range(20):
  predict = predictions[i].tolist()
  with open(seq2text(X_[i]) + 'predict.json', 'w', encoding='utf-8') as make_file:
    json.dump(predict, make_file, indent="\t")

for i in range(20):
  predict = y[i].tolist()
  with open(seq2text(X_[i]) + 'original.json', 'w', encoding='utf-8') as make_file:
    json.dump(predict, make_file, indent="\t")
