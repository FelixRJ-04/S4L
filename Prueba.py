import cv2
import mediapipe as mp

#Inicialización de MediaPipe con sus utilidades para dibujo
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

#Inicialización de captura de video
cap = cv2.VideoCapture(0)

#Configuración del modelo Holistic
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignorando el frame vacio de la camara.")
      continue

    #Se prepara la imagen para el procesamiento
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # Conversión de imagen a BGR para su muestreo
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #Detección de rostro
    if results.face_landmarks:
      mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec = None,
        connection_drawing_spec = mp_drawing_styles
        .get_default_face_mesh_contours_style())
      
    
    #Detección y dibujo del cuerpo (pose)
    if results.pose_landmarks:
      mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec = mp_drawing_styles
        .get_default_pose_landmarks_style())

    #Detección y dibujo de las manos
    if results.right_hand_landmarks:
      mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))
      
    if results.left_hand_landmarks:
      mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))
      

    # Muestreo de camara con modelos ya cargados
    cv2.imshow('MediaPipe Holistic y Hand Tracker', image)

    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
cv2.destroyAllWindows()
hola