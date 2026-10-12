import cv2
import pandas as pd
import datetime
import os
import pickle

model_path = "models/face_model.yml"
labels_path = "models/labels.pkl"
attendance_file = "attendance/attendance.csv"

# Load recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

# Load label map
with open(labels_path, "rb") as f:
    label_map = pickle.load(f)

# Reverse the dictionary (ID → Name)
id_to_name = {v: k for k, v in label_map.items()}

# Haar cascade for face detection
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Ensure attendance folder exists
if not os.path.exists("attendance"):
    os.makedirs("attendance")

# Load existing attendance or create new
if os.path.exists(attendance_file):
    df = pd.read_csv(attendance_file)
else:
    df = pd.DataFrame(columns=["Name", "Date", "Time"])

cap = cv2.VideoCapture(0)

present_people = set()

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    current_frame_people = set()

    for (x, y, w, h) in faces:
        id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if conf < 50:  # stricter threshold
            name = id_to_name.get(id_, f"User_{id_}")
            current_frame_people.add(name)

            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

            # Log Present if not already logged
            if name not in present_people:
                present_people.add(name)
                now = datetime.datetime.now()
                df = pd.concat([df, pd.DataFrame([[name, now.strftime("%Y-%m-%d"), "Present"]], columns=df.columns)], ignore_index=True)
                df.to_csv(attendance_file, index=False)

    # Detect who left
    absent_people = present_people - current_frame_people
    for person in absent_people:
        present_people.remove(person)
        now = datetime.datetime.now()
        df = pd.concat([df, pd.DataFrame([[person, now.strftime("%Y-%m-%d"), "Absent"]], columns=df.columns)], ignore_index=True)
        df.to_csv(attendance_file, index=False)

    cv2.imshow("Face Recognition Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()