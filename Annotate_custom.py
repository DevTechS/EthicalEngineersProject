import csv
import os
import cv2
import time
import face_recognition
import numpy as np

use_percent = 100

colors = {
    'chin': (0, 0, 255),          # Red
    'left_eyebrow': (0, 255, 0),  # Green
    'right_eyebrow': (255, 0, 0), # Blue
    'nose_bridge': (0, 255, 255), # Yellow
    'nose_tip': (255, 255, 0),    # Cyan
    'left_eye': (255, 0, 255),    # Magenta
    'right_eye': (0, 165, 255),   # Orange
    'top_lip': (128, 0, 128),     # Purple
    'bottom_lip': (128, 128, 0)   # Teal
}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def balance_bright(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    img_brightness = np.median(v)
    brightness_change = round((128 - img_brightness)/3)

    if brightness_change > 0:
        lim = 255 - np.uint8(brightness_change)
        v[v <= lim] += np.uint8(brightness_change)
        v[v > lim] = 255
    if brightness_change < 0:
        darken = np.uint8(-1*brightness_change)
        v[v <= darken] = 0
        v[v > darken] -= darken
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def Process_IMG(file_path):
    image = cv2.imread(file_path)
    blurred_image = cv2.GaussianBlur(image, (0, 0), 10)
    sharp_image = cv2.addWeighted(image, 1.5, blurred_image, -0.5, 0)
    bright_image = balance_bright(image)

    working_image = image

    face_locations = face_recognition.face_locations(working_image)
    if len(face_locations) == 0: # try sharpening the image
        working_image = sharp_image
        face_locations = face_recognition.face_locations(working_image)
    if len(face_locations) == 0: # try brightening the image
        working_image = bright_image
        face_locations = face_recognition.face_locations(working_image)

    face_count = len(face_locations)
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)

    face_landmarks_list = face_recognition.face_landmarks(working_image)
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            for point in face_landmarks[facial_feature]:
                cv2.circle(image, point, 2, colors[facial_feature], -1)
                
    if len(face_locations) == 0: # If all else fails, CascadeClassifier fallback
        faces = face_cascade.detectMultiScale(sharp_image, scaleFactor=1.1, minNeighbors=2, minSize=(140, 140))
        face_count = len(faces)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    cv2.imwrite("done_custom_" + file_path, image)
    return face_count

def load_csv_as_2d_array(csv_file_path):
    data = []
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    return data

csv_file_path = "fairface_label_train.csv"
csv_data = load_csv_as_2d_array(csv_file_path)
if os.path.exists('output_custom.csv'):
        os.remove('output_custom.csv')
with open('output_custom.csv', 'w') as file:
    file.write('file,age,gender,race,faces\n')
    rows = 0
    start_time = time.time()
    print("Task Started. Please be patient.")
    for row in csv_data[1:]:
        rows += 1
        if (rows%round(len(csv_data)/100) == 0):
            progress = rows/len(csv_data)
            print(round(100*progress,1), "%")
            current_time = time.time()
            passed_minutes = (current_time-start_time)/60
            print("Time Taken:", round(passed_minutes,1),'m')
            print("Time Remaining:", round((passed_minutes/progress - passed_minutes),1),'m')
        features = Process_IMG(row[0]) # faces
        file.write(f'{row[0]},{row[1]},{row[2]},{row[3]},{features}\n')
        if rows/len(csv_data) > use_percent/100.0:
            break
print("End Condition Reached!")