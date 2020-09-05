import pandas as pd 
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import os
"""tf.config.experimental.set_memory_growth(
     tf.config.list_physical_devices('GPU'), enable=True
)"""

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
callbacks = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_freq=2)
data = pd.read_csv("t_bbe.csv")
sentences = data["t"]
sentences = np.array(sentences)
sentences.reshape(1,-1)
sentences = sentences.tolist()
text = ' '.join([i for i in sentences])
chars = sorted(list(set(text)))
print(enumerate(chars))
n_2_chr = dict((i, c) for i, c in enumerate(chars))
chr_2_n = dict((c, i) for i, c in enumerate(chars)) #analyticsvidhya.com

data_train = []
next_char = []
seq_length = 100

for i in range(0, len(text)-seq_length):
    s,l = text[i:i+seq_length], text[i+seq_length]
    data_train.append([chr_2_n[char] for char in s])
    next_char.append(chr_2_n[l])

x_train = np.reshape(data_train, (len(data_train), seq_length, 1))
x_train = x_train / float(len(chars))

y = to_categorical(next_char)
print(x_train.shape)
def create_model():
    return (Sequential([

        LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True),
        LSTM(32),
        Dropout(0.2),
        Dense(y.shape[1], activation='softmax')]))
model = create_model()


model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])
#model.fit(x_train, y, epochs=7, batch_size=16)
#model.save_weights(checkpoint_path)
model.load_weights(checkpoint_path)

string_mapped = data_train[np.random.randint(0, len(data_train) - 1)]
last_string = [n_2_chr[value] for value in string_mapped]
for i in range(seq_length):
    x = np.reshape(string_mapped, (1, len(string_mapped),1))
    x = x / float(len(chars))
    pred_index = np.argmax(model.predict(x, verbose=0))
    seq = [n_2_chr[value] for value in string_mapped]
    string_mapped.append(pred_index)
    string_mapped = string_mapped[1:len(string_mapped)]

txt = ""
for char in last_string:
    txt = txt+char
print(txt)