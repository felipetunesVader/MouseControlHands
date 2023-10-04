import cv2
import mediapipe as mp
import pyautogui

# Inicializar o módulo de mãos do mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Inicializar o módulo de desenho do mediapipe
mp_drawing = mp.solutions.drawing_utils

# Capturar vídeo da webcam
cap = cv2.VideoCapture(0)

# Obter as dimensões da tela
screen_width, screen_height = pyautogui.size()

# Constante para determinar a proximidade entre o polegar e o indicador para um clique
CLICK_THRESHOLD = 40  # Ajuste conforme necessário

# Variável para rastrear o estado anterior do clique (para evitar cliques múltiplos)
previous_click_state = False

while cap.isOpened():
    ret, frame = cap.read()

    # Converter a imagem BGR do OpenCV para RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processar a imagem e obter os resultados
    results = hands.process(image_rgb)

    # Se uma mão for detectada
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Obter a posição do pulso
            cx, cy = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1]), \
                int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * frame.shape[0])

            # Mover o mouse para a posição correspondente na tela
            pyautogui.moveTo(screen_width - (cx * screen_width / frame.shape[1]),
                             cy * screen_height / frame.shape[0])

            # Obter a posição do polegar e do indicador
            thumb = [int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1]),
                     int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0])]
            index = [int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]),
                     int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])]

            # Calcular a distância entre o polegar e o indicador
            distance = ((thumb[0] - index[0]) ** 2 + (thumb[1] - index[1]) ** 2) ** 0.5

            # Se a distância for menor que o limiar e o clique anterior não estiver ativo, interprete como um clique
            if distance < CLICK_THRESHOLD and not previous_click_state:
                pyautogui.click()
                previous_click_state = True
            elif distance >= CLICK_THRESHOLD:
                previous_click_state = False

            # Desenhar os landmarks da mão na imagem
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Mouse Control", frame)

    # Se a tecla 'q' for pressionada, sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
