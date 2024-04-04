import csv
import os
import cv2
import time
import face_recognition

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

def Process_IMG(file_path):
    image = cv2.imread(file_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)

    face_count = len(face_locations)
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)

    face_landmarks_list = face_recognition.face_landmarks(rgb_image)
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            for point in face_landmarks[facial_feature]:
                cv2.circle(image, point, 2, colors[facial_feature], -1)

    cv2.imwrite("done_face_recognition_" + file_path, image)
    return face_count, face_count

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
delete_file_if_exists('output_face_recognition.csv')
with open('output_face_recognition.csv', 'w') as file:
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
        file.write(f'{row[0]},{row[1]},{row[2]},{row[3]},{features[0]}\n')
