from __future__ import print_function
import os
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score)
from sklearn.preprocessing import Normalizer
from keras.models import load_model

# Carregar os dados
traindata = pd.read_csv(os.path.join('kdd', 'binary', 'Training.csv'), header=None)
testdata = pd.read_csv(os.path.join('kdd', 'binary', 'Testing.csv'), header=None)

X = traindata.iloc[:, 1:42]
Y = traindata.iloc[:, 0]
C = testdata.iloc[:, 0]
T = testdata.iloc[:, 1:42]

# Normalizar os dados
scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

y_train = np.array(Y)
y_test = np.array(C)

X_train = np.array(trainX)
X_test = np.array(testT)

batch_size = 64

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

# Avaliar o modelo carregando os pesos salvos
# Listar arquivos de pesos compatíveis
weight_files = [f for f in os.listdir(os.path.join("kddresults", "dnn3layer")) if f.endswith(('.keras', '.weights.h5', '.h5'))]

score = []
name = []

for file in weight_files:
    model = load_model(os.path.join("kddresults", "dnn3layer", file))
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="binary")
    precision = precision_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")
    
    print("----------------------------------------------")
    print(f"File: {file}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    score.append(accuracy)
    name.append(file)

# Carregar o melhor modelo e fazer previsões finais
model.load_weights(os.path.join("kddresults", "dnn3layer", name[score.index(max(score))]))
pred = (model.predict(X_test) > 0.5).astype("int32")
proba = model.predict(X_test)

# Salvar as previsões e probabilidades
np.savetxt(os.path.join("dnnres", "dnn3predicted.txt"), pred)
np.savetxt(os.path.join("dnnres", "dnn3probability.txt"), proba)

# Calcular e exibir as métricas finais
accuracy = accuracy_score(y_test, pred)
recall = recall_score(y_test, pred, average="binary")
precision = precision_score(y_test, pred, average="binary")
f1 = f1_score(y_test, pred, average="binary")

print("----------------------------------------------")
print("accuracy")
print("%.3f" % accuracy)
print("precision")
print("%.3f" % precision)
print("recall")
print("%.3f" % recall)
print("f1score")
print("%.3f" % f1)
