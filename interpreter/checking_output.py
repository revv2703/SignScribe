import time
from interpreter.data_preprocessing import *
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import pygame.mixer
from gtts import gTTS
from interpreter.mic_input_handler import play

json_file = open("interpreter/model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("interpreter/model.h5")

colors = []
for i in range(0,20):
    colors.append((245,117,16))
print(len(colors))

def prob_viz(res, actions, input_frame, colors,threshold):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


# New detection variables

# def real_time_asl():
#     sentence = []
#     sequence = []
#     accuracy = []
#     predictions = []
#     threshold = 0.8
#     cap = cv2.VideoCapture(0)

#     with mp_hands.Hands(
#         model_complexity=0,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     ) as hands:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             cropframe = frame[40:400, 0:300]
#             frame = cv2.rectangle(frame, (0, 40), (300, 400), (0, 255, 0), 2)
#             image, results = mediapipe_detection(cropframe, hands)

#             # Prediction logic
#             keypoints = extract_keypoints(results)
#             sequence.append(keypoints)
#             sequence = sequence[-30:]

#             try:
#                 if len(sequence) == 30:
#                     res = model.predict(np.expand_dims(sequence, axis=0))[0]
#                     print(actions[np.argmax(res)])
#                     predictions.append(np.argmax(res))

#                     if np.unique(predictions[-10:])[0] == np.argmax(res):
#                         if res[np.argmax(res)] > threshold:
#                             if len(sentence) > 0:
#                                 if actions[np.argmax(res)] != sentence[-1]:
#                                     # Check for new addition to sentence before triggering text-to-speech
#                                     sentence.append(actions[np.argmax(res)])
#                                     accuracy.append(str(res[np.argmax(res)] * 100))
#                                     text_to_speech(' '.join(sentence))  # Trigger text-to-speech only on new addition
#                             else:
#                                 sentence.append(actions[np.argmax(res)])
#                                 accuracy.append(str(res[np.argmax(res)] * 100))
#                                 text_to_speech(' '.join(sentence))  # Trigger text-to-speech only on new addition

#             except Exception as e:
#                 pass

#             cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 255), -1)
#             cv2.putText(frame, ' '.join(sentence), (3, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#             print(sentence)

#             cv2.imshow('OpenCV Feed', frame)

#             if cv2.waitKey(10) & 0xFF == ord('q'):
#                 break

#     cap.release()
#     cv2.destroyAllWindows()

def real_time_asl():
    sentence = []
    sequence = []
    accuracy = []
    predictions = []
    threshold = 0.8
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            cropframe = frame[40:400, 0:300]
            frame = cv2.rectangle(frame, (0, 40), (300, 400), (0, 255, 0), 2)
            image, results = mediapipe_detection(cropframe, hands)

            # Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            try:
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    predictions.append(np.argmax(res))

                    if np.unique(predictions[-10:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > threshold:
                            if len(sentence) > 0:
                                if actions[np.argmax(res)] != sentence[-1]:
                                    # Only speak the newly added character
                                    text_to_speech(actions[np.argmax(res)])
                                    sentence.append(actions[np.argmax(res)])
                                    accuracy.append(str(res[np.argmax(res)] * 100))
                            else:
                                sentence.append(actions[np.argmax(res)])
                                accuracy.append(str(res[np.argmax(res)] * 100))
                                text_to_speech(actions[np.argmax(res)])  # Speak the first character

            except Exception as e:
                pass

            cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 255), -1)
            cv2.putText(frame, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            print(sentence)

            cv2.imshow('OpenCV Feed', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
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
    # time.sleep(1)
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    # Release the audio file
    pygame.mixer.music.unload()
    # time.sleep(1)


def play_audio(output):
    global lock
    if lock.acquire(blocking=False):
        play(output)
        lock.release()

if __name__ == "__main__":
    real_time_asl()