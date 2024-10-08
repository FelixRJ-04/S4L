import os 
import pandas as pd
from mediapipe.python.solutions.holistic import Holistic
from Dibuja import *
from Constantes import *

def creaCarpeta(ruta):
    if not os.path.exists(ruta):
        os.makedirs(ruta)

def guardaKeypoints(IDPalabra, rutaPalabra, rutaHDF):
    data = pd.DataFrame([])
    rutaFrame = os.path.join(rutaPalabra, IDPalabra)
    
    with Holistic() as holistic:
        print(f'Creando keypoints de "{IDPalabra}"...')
        ListaMuestras = os.listdir(rutaFrame)
        CuentaMuestras = len(ListaMuestras)
        
        for numMuest, nomMuestra in enumerate(ListaMuestras, start=1):
            rutaMuestra = os.path.join(rutaFrame, nomMuestra)
            keypoints_sequence = ObtenKeypoints(holistic, rutaMuestra)
            data = InsertaSecuenciaKeypoints(data, numMuest, keypoints_sequence)
            print(f"{numMuest}/{CuentaMuestras}", end="\r")
            
    data.to_hdf(rutaHDF, key='data', mode='w')
    print(f"Keypoints creados! ({CuentaMuestras} muestras)", end="\n")

if __name__ == "__main__":
    # Crea la carpeta `keypoints` en caso no exista
    creaCarpeta(RutaKeypoints)
    IDPalabras = [word for word in os.listdir(os.path.join(RutaRaiz, RutaFrameActions))]
    for IDPalabra in IDPalabras:
        rutaHDF = os.path.join(RutaKeypoints, f"{IDPalabra}.h5")
        guardaKeypoints(IDPalabra, RutaFrameActions, rutaHDF)