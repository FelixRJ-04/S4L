import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from keras.models import load_model
from Dibuja import *
from Constantes  import *
from Muestras import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gtts import gTTS
import os
import pygame
from time import sleep

def text_to_speech(text):
    tts = gTTS(text=text, lang='es')
    filename = "speech.mp3"
    tts.save(filename)
    
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        sleep(1)

    pygame.mixer.quit()
    pygame.quit()

    os.remove(filename)
    
def interpolate_keypoints(keypoints, target_length=15):
    current_length = len(keypoints)
    if current_length == target_length:
        return keypoints
    
    indices = np.linspace(0, current_length - 1, target_length)
    interpolated_keypoints = []
    for i in indices:
        lower_idx = int(np.floor(i))
        upper_idx = int(np.ceil(i))
        weight = i - lower_idx
        if lower_idx == upper_idx:
            interpolated_keypoints.append(keypoints[lower_idx])
        else:
            interpolated_point = (1 - weight) * np.array(keypoints[lower_idx]) + weight * np.array(keypoints[upper_idx])
            interpolated_keypoints.append(interpolated_point.tolist())
    
    return interpolated_keypoints

def normalize_keypoints(keypoints, target_length=15):
    current_length = len(keypoints)
    if current_length < target_length:
        return interpolate_keypoints(keypoints, target_length)
    elif current_length > target_length:
        step = current_length / target_length
        indices = np.arange(0, current_length, step).astype(int)[:target_length]
        return [keypoints[i] for i in indices]
    else:
        return keypoints
    
def evaluate_model(src=None, threshold=0.8, margin_frame=1, delay_frames=3):
    kp_seq, sentence = [], []
    contenido=os.listdir('./data_5M/keypoints')
    contenido = np.array(contenido, dtype=str)
    word_ids =[]
    for NumPalabra, Palabra in enumerate(contenido):
        Palabra = contenido[NumPalabra]
        Palabra= Palabra.replace(".h5", "")
        word_ids.append(Palabra)
            
    model = load_model('./Modelo_Ale.h5')
    count_frame = 0
    fix_frames = 0
    recording = False
    
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(src or 0)
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret: break

            results = mediapipe_detection(frame, holistic_model)
            
            # TODO: colocar un máximo de frames para cada seña,
            # es decir, que traduzca incluso cuando hay mano si se llega a ese máximo.
            if hayMano(results) or recording:
                recording = False
                count_frame += 1
                if count_frame > margin_frame:
                    kp_frame = extract_keypoints(results)
                    kp_seq.append(kp_frame)
            
            else:
                if count_frame >= MIN_LENGTH_FRAMES + margin_frame:
                    fix_frames += 1
                    if fix_frames < delay_frames:
                        recording = True
                        continue
                    kp_seq = kp_seq[: - (margin_frame + delay_frames)]
                    kp_normalized = normalize_keypoints(kp_seq, int(MODEL_FRAMES))
                    res = model.predict(np.expand_dims(kp_normalized, axis=0))[0]
                    
                    print(np.argmax(res), f"({res[np.argmax(res)] * 100:.2f}%)")
                    print(word_ids[np.argmax(res)])
                    if res[np.argmax(res)] > threshold:
                        sent = word_ids[np.argmax(res)]
                        print(word_ids)
                        sentence.insert(0, sent)
                        text_to_speech(sent) # ONLY LOCAL (NO SERVER)
                
                recording = False
                fix_frames = 0
                count_frame = 0
                kp_seq = []
            
            if not src:
                cv2.rectangle(frame, (0, 0), (640, 35), (245, 117, 16), -1)
                cv2.putText(frame, ' | '.join(sentence), FONT_POS, FONT, FONT_SIZE, (255, 255, 255))
                
                draw_keypoints(frame, results)
                cv2.imshow('Traductor LSP', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
        video.release()
        cv2.destroyAllWindows()
        return sentence
    
if __name__ == "__main__":
    evaluate_model()