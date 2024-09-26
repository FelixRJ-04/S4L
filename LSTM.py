#Establecer librerias para el entrenamiendo de la red neuronal
import numpy as np #Numpy se utiliza para reconocer los arreglos y poder manipularlos para su entrenamiento
import h5py #H5py nos sirve para abrir los archivos tipo .h5 y de esta forma poder obtener sus datos
import tensorflow #Nos ayuda a realizar el proceso del entrenamiento del modelo con sus diferentes librerias

from keras.models import Sequential  #Modelos, capas y utilidades 
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

path= "./data/keypoints/Adios.h5"

def load_data(path):
    with h5py.File(path, 'r') as f:
        X= np.array(f['X'])
        y= np.array(f['y'])

        return X, y
    
def preprocess_data(X, y, test_size= 0.2, random_state= 42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, X_test, y_train, y_test

def create_lstm_model(timesteps, n_features, n_clases):
    model = Sequential
    model.add(LSTM(units=64, return_sequences=True, input_shape=(timesteps, n_features)))
    model.add(LSTM(units=64))
    model.add(Dense(units=n_clases, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acurracy'])

    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    train = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return train

X, y = load_data(path)

n_samples, timesteps, n_features = X.shape
n_classes = len(np.unique(y))

X_train, X_test, y_train, y_test = preprocess_data(X, y)

model = create_lstm_model(timesteps, n_features, n_classes)

train = train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32)

model.save('Modelo_Prueba_LSTM.h5')

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Perdida: {loss}, Precisión: {accuracy}")
