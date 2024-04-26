import threading
from interpreter.mic_input_handler import play
from interpreter.src.backbone import TFLiteModel, get_model
from interpreter.src.landmarks_extraction import mediapipe_detection, draw, extract_coordinates, load_json_file 
from interpreter.src.config import SEQ_LEN, THRESH_HOLD
import numpy as np
import cv2
import time
import mediapipe as mp
from gtts import gTTS
import os
import pyaudio
import pygame.mixer

mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils

s2p_map = {k.lower():v for k,v in load_json_file("interpreter/src/sign_to_prediction_index_map.json").items()}
p2s_map = {v:k for k,v in load_json_file("interpreter/src/sign_to_prediction_index_map.json").items()}
encoder = lambda x: s2p_map.get(x.lower())
decoder = lambda x: p2s_map.get(x)

models_path = [
    'interpreter\models\islr-fp16-192-8-seed_all42-foldall-last.h5',
]
models = [get_model() for _ in models_path]

# Load weights from the weights file.
for model,path in zip(models,models_path):
    model.load_weights(path)
    
lock = threading.Lock()

def real_time_asl():
    """
    Perform real-time ASL recognition using webcam feed.

    This function initializes the required objects and variables, captures frames from the webcam, processes them for hand tracking and landmark extraction, and performs ASL recognition on a sequence of landmarks.

    Args:
        None

    Returns:
        None
    """
    res = []
    tflite_keras_model = TFLiteModel(islr_models=models)
    sequence_data = []
    cap = cv2.VideoCapture(0)
    
    start = time.time()
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # The main loop for the mediapipe detection.
        while cap.isOpened():
            ret, frame = cap.read()
            
            start = time.time()
            
            image, results = mediapipe_detection(frame, holistic)
            draw(image, results)
            
            try:
                landmarks = extract_coordinates(results)
            except:
                landmarks = np.zeros((468 + 21 + 33 + 21, 3))
            sequence_data.append(landmarks)
            
            sign = ""
            
            # Generate the prediction for the given sequence data.
            if len(sequence_data) % SEQ_LEN == 0:
                prediction = tflite_keras_model(np.array(sequence_data, dtype = np.float32))["outputs"]

                if np.max(prediction.numpy(), axis=-1) > THRESH_HOLD:
                    sign = np.argmax(prediction.numpy(), axis=-1)
                
                sequence_data = []
            
            image = cv2.flip(image, 1)
            
            cv2.putText(image, f"{len(sequence_data)}", (3, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            image = cv2.flip(image, 1)
            
            # Insert the sign in the result set if sign is not empty.
            # if sign != "" and decoder(sign) not in res:
            if sign != "":
                res.insert(0, decoder(sign))
                print(res)
                text_to_speech(' '.join(res))
            
            # Get the height and width of the image
            height, width = image.shape[0], image.shape[1]

            # Create a white column
            white_column = np.ones((height // 8, width, 3), dtype='uint8') * 255

            # Flip the image vertically
            image = cv2.flip(image, 1)
            
            # Concatenate the white column to the image
            image = np.concatenate((white_column, image), axis=0)
            
            cv2.putText(image, f"{', '.join(str(x) for x in res)}", (3, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2, cv2.LINE_AA)
                            
            cv2.imshow('Webcam Feed',image)
            
            # Wait for a key to be pressed.
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        

def text_to_speech(text, lang='en', filename='output.mp3'):
    """
    Convert text to speech and save as an audio file.

    Parameters:
    text (str): The text to convert to speech.
    lang (str): The language of the text (default is 'en' for English).
    filename (str): The filename to save the audio file (default is 'output.mp3').

    Returns:
    None
    """
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)
    print(f"Text converted to speech and saved as {filename}")
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()


def play_audio(output):
    global lock
    if lock.acquire(blocking=False):
        play(output)
        lock.release()
        

if __name__ == "__main__":
    real_time_asl()