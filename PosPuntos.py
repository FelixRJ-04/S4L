import os 
import pandas as pd
from mediapipe.python.solutions.holistic import Holistic
from Keypoints import get_keypoints, insert_keypoints_sequence
from Constantes import *

def creaCarpeta(path):
    if not os.path.exists(path):
        os.makedirs(path)

def guardaKeypoints(word_id, words_path, hdf_path):
    data = pd.DataFrame([])
    frames_path = os.path.join(words_path, word_id)
    
    with Holistic() as holistic:
        print(f'Creando keypoints de "{word_id}"...')
        sample_list = os.listdir(frames_path)
        sample_count = len(sample_list)
        
        for n_sample, sample_name in enumerate(sample_list, start=1):
            sample_path = os.path.join(frames_path, sample_name)
            keypoints_sequence = get_keypoints(holistic, sample_path)
            data = insert_keypoints_sequence(data, n_sample, keypoints_sequence)
            print(f"{n_sample}/{sample_count}", end="\r")
            
    data.to_hdf(hdf_path, key='data', mode='w')
    print(f"Keypoints creados! ({sample_count} muestras)", end="\n")

if __name__ == "__main__":
    # Crea la carpeta `keypoints` en caso no exista
    creaCarpeta(KEYPOINTS_PATH)
    word_ids = [word for word in os.listdir(os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH))]
    for word_id in word_ids:
        hdf_path = os.path.join(KEYPOINTS_PATH, f"{word_id}.h5")
        guardaKeypoints(word_id, FRAME_ACTIONS_PATH, hdf_path)