import csv
import os
import cv2
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def Process_IMG(file_path):
    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))
    face_count = len(faces)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    eyes = eye_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    eye_count = len(eyes)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(image, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    mouths = mouth_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    mouth_count = len(mouths)
    for (mx, my, mw, mh) in mouths:
        cv2.rectangle(image, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)

    cv2.imwrite("done_cv2_" + file_path, image)
    return face_count, eye_count, mouth_count

def load_csv_as_2d_array(csv_file_path):
    data = []
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    return data

def delete_file_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


csv_file_path = "fairface_label_train.csv"
csv_data = load_csv_as_2d_array(csv_file_path)
delete_file_if_exists('output_cv2.csv')
with open('output_cv2.csv', 'w') as file:
    file.write('file,age,gender,race,faces,eyes,mouths\n')
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
        features = Process_IMG(row[0]) # faces,eyes,mouths
        file.write(f'{row[0]},{row[1]},{row[2]},{row[3]},{features[0]},{features[1]},{features[2]}\n')
