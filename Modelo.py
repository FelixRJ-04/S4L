import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Masking, Bidirectional, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def extrae_datos():
    contenido=os.listdir('./data_5M/keypoints')
    contenido = np.array(contenido, dtype=str)
    secuencia, etiqueta =[],  []
    for NumPalabra, Palabra in enumerate(contenido):
        Carpeta = contenido[NumPalabra]
        ruta = os.path.join('./data_5M/keypoints', f"{Carpeta}")
        data = pd.read_hdf(ruta, key='data')
        for _, imMues in data.groupby('sample'):
            secDePuntos= [fila['keypoints']for _, fila in imMues.iterrows()]
            secuencia.append(secDePuntos)
            Palabra_limpia = Palabra.replace(".h5", "").replace("1", "").replace("2", "").replace("3", "")
            etiqueta.append(NumPalabra)
    return secuencia, etiqueta
contenido=[]
sec, etiqueta = extrae_datos()

secuencia = pad_sequences(sec, 30, padding='post', truncating='post', dtype='float32', value=0.0)
numClases=len(set(etiqueta))
X=np.array(secuencia)
y = to_categorical(etiqueta, num_classes=numClases).astype(int)
print(f"Forma de X: {X.shape}")
print(f"Forma de Y: {y.shape}")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = Sequential()
modelo.add(Masking(mask_value=0.0, input_shape=(30, 1662)))
modelo.add(Bidirectional(LSTM(64, return_sequences=True)))
modelo.add(Dropout(0.4))
modelo.add(Bidirectional(LSTM(64, return_sequences=True)))
modelo.add(Dropout(0.4))
modelo.add(Bidirectional(LSTM(64, return_sequences=True)))
modelo.add(Dropout(0.4))
modelo.add(Bidirectional(LSTM(64, return_sequences=True)))
modelo.add(Dropout(0.4))
modelo.add(LSTM(128))
modelo.add(Dense(numClases, activation='softmax'))

optimizer = Adam(learning_rate=0.0001)
modelo.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
historial = modelo.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_val, y_val))

resultados = modelo.evaluate(X_val, y_val, verbose=0)
print(f'Perdida en validacion: {resultados[0]:.4f}')
print(f'Precision en validacion: {resultados[1]*100:.2f}%')

y_pred = modelo.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

modelo.save('./Modelo_Felix.h5')
cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(xticks_rotation='vertical')
plt.show()