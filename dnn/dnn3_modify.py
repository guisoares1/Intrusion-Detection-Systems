from __future__ import print_function
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
np.random.seed(1337)  # para reprodutibilidade
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils import to_categorical
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score, mean_squared_error, mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.optimizers import Adam

# Criar diretório se ele não existir
output_dir = os.path.join('kddresultsModify', 'dnn3layer')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

traindata = pd.read_csv(os.path.join('kdd', 'binary', 'Training.csv'), header=None)
testdata = pd.read_csv(os.path.join('kdd', 'binary', 'Testing.csv'), header=None)

X = traindata.iloc[:, 1:42]
Y = traindata.iloc[:, 0]
C = testdata.iloc[:, 0]
T = testdata.iloc[:, 1:42]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

y_train = np.array(Y)
y_test = np.array(C)

X_train = np.array(trainX)
X_test = np.array(testT)

batch_size = 64

# Verificar se existe um checkpoint salvo
checkpoint_dir = os.path.join('kddresultsModify', 'dnn3layer')
checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint')]
if checkpoints:
    # Carregar o último checkpoint salvo
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1].split('.')[0]))
    model = load_model(os.path.join(checkpoint_dir, latest_checkpoint))
    initial_epoch = int(latest_checkpoint.split('-')[1].split('.')[0])
    print(f"Carregando o modelo do checkpoint: {latest_checkpoint}")
else:
    # Definir a rede neural
    model = Sequential()
    model.add(Dense(1024, input_dim=41, activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(768, activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    # Compilar o modelo
    optimizer = Adam(learning_rate=0.015)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Callbacks
checkpointer = ModelCheckpoint(filepath=os.path.join('kddresultsModify', 'dnn3layer', 'checkpoint-{epoch:02d}.keras'), verbose=1, save_best_only=True, monitor='loss')
csv_logger = CSVLogger(os.path.join('kddresultsModify', 'dnn3layer', 'training_set_dnnanalysis.csv'), separator=',', append=False)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001, verbose=1)

# Treinar o modelo com checkpoints
model.fit(X_train, y_train, batch_size=batch_size, epochs=1000, callbacks=[checkpointer, csv_logger])

model.save(os.path.join('kddresultsModify', 'dnn3layer', 'dnn3layer_model.keras'))
