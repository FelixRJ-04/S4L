import numpy as np
import pandas as pd
import tensorflow as tf
import os
import random

from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Dropout
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

base_path = "./data/keypoints"

def extrae_datos(base_path):
    secuencias, etiquetas = [], []
    carpetas = os.listdir(base_path)
    random.shuffle(carpetas)

    for carpeta in carpetas:
        ruta_carpeta = os.path.join(base_path, carpeta)
        if carpeta.endswith(".h5"):
            data = pd.read_hdf(ruta_carpeta)

        for _, imMues in data.groupby('sample'):
            secuencia_puntos = [fila['keypoints'] for _, fila in imMues.iterrows()]
            random.shuffle(secuencia_puntos)
            secuencias.append(secuencia_puntos)
            etiqueta = carpeta.replace(".h5", "")
            etiquetas.append(etiqueta)

    return np.array(secuencias), np.array(etiquetas)

def load_and_preprocess_data(base_path, test_size=0.2):
    secuencias, etiquetas = extrae_datos(base_path)

    print(f"Total de muestras cargadas: {len(secuencias)}")

    X_train, X_test, y_train, y_test = train_test_split(secuencias, etiquetas, test_size=test_size, random_state=42)

    etiquetas_unicas = np.unique(y_train)
    num_clases = len(etiquetas_unicas)

    print(f"Clases unicas: {etiquetas_unicas}")
    print(f"Número de clases: {num_clases}")

    y_train_cat = to_categorical([np.where(etiquetas_unicas == y)[0][0] for y in y_train], num_classes=num_clases)
    y_test_cat = to_categorical([np.where(etiquetas_unicas == y)[0][0] for y in y_test], num_classes=num_clases)

    timesteps = len(X_train[0])
    n_features = len(X_train[0][0])

    X_train = X_train.reshape((X_train.shape[0], timesteps, n_features))
    X_test = X_test.reshape((X_test.shape[0], timesteps, n_features))

    return X_train, X_test, y_train_cat, y_test_cat, timesteps, n_features, num_clases

def create_lstm_model(timesteps, n_features, n_clases):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=128, return_sequences=True), input_shape=(timesteps, n_features)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(units=128)))
    model.add(Dropout(0.3))
    model.add(Dense(units=n_clases, activation='softmax', kernel_regularizer=l2(0.001)))

    model.summary()

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=200, batch_size=64):
    train = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return train

X_train, X_test, y_train_cat, y_test_cat, timesteps, n_features, n_clases = load_and_preprocess_data(base_path)

model = create_lstm_model(timesteps, n_features, n_clases)

train_model(model, X_train, y_train_cat, X_test, y_test_cat, epochs=200, batch_size=64)

model.save('./Modelo_Prueba_LSTM.h5')

loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f"Perdida: {loss}, Precisión: {accuracy}")