import cv2
import numpy as np
import threading
import time
import os
import speech_recognition as sr
from PIL import Image

# Constants
image_x, image_y = 64, 64
op_dest = "composer/filtered_data/"
alpha_dest = "composer/alphabet/"
model_path = 'model.h5'


# Global variables
dirListing = os.listdir(op_dest)
gif_frames = []
save_lock = threading.Lock()
editFiles = []
for item in dirListing:
       if ".webp" in item:
              editFiles.append(item)

file_map={}
for i in editFiles:
    tmp=i.replace(".webp","")
    print(tmp)
    tmp=tmp.split()
    file_map[i]=tmp

# Utility functions
def give_char(result):
    chars = "ABCDEFGHIJKMNOPQRSTUVWXYZ"
    indx = np.argmax(result[0])
    return chars[indx]

def check_sim(i, file_map):
    for item in file_map:
        for word in file_map[item]:
            if i == word:
                return 1, item
    return -1, ""

def func(a):
    global gif_frames
    all_frames = []
    final = Image.new('RGB', (380, 260))
    words = a.split()
    for i in words:
        flag, sim = check_sim(i, file_map)
        if flag == -1:
            for j in i:
                im = Image.open(alpha_dest + str(j).lower() + "_small.gif")
                frameCnt = im.n_frames
                for frame_cnt in range(frameCnt):
                    im.seek(frame_cnt)
                    all_frames.append(im.convert('RGB'))
        else:
            im = Image.open(op_dest + sim)
            im.info.pop('background', None)
            im.save('tmp.gif', 'gif', save_all=True)
            im = Image.open("tmp.gif")
            frameCnt = im.n_frames
            for frame_cnt in range(frameCnt):
                im.seek(frame_cnt)
                all_frames.append(im.convert('RGB'))
    final.save("out.gif", save_all=True, append_images=all_frames, duration=500, loop=0)
    gif_frames = all_frames

def hear_voice():
    store = sr.Recognizer()
    with sr.Microphone() as s:
        print("Speak now...")
        audio_input = store.record(s, duration=5)
        try:
            text_output = store.recognize_google(audio_input)
            print(text_output)
            return text_output
            # return "any"
        except Exception as e:
            print("Error Hearing Voice:", e)
            return ''

# Thread function to continuously save GIF frames
def save_gif():
    global gif_frames
    while True:
        time.sleep(3)  # Wait for 3 seconds before saving the next GIF
        if gif_frames:
            save_lock.acquire()
            try:
                final = Image.new('RGB', (380, 260))
                final.save("out.gif", save_all=True, append_images=gif_frames, duration=100, loop=0)
            finally:
                save_lock.release()

# Main function
def main():
    global gif_frames
    save_thread = threading.Thread(target=save_gif)
    save_thread.start()
    print("Welcome to Two Way Sign Language Translator CLI!")
    print("\nChoose an option:")
    print("1. Voice to Sign")
    print("2. Exit")
    choice = '1'
    while True:
        text_input = hear_voice()
        if text_input:
            func(text_input)
        

# Run main function if the file is executed directly
if __name__ == "__main__":
    main()
