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

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if conf < 70:  # confidence threshold
            name = id_to_name.get(id_, f"User_{id_}")

            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            now = datetime.datetime.now()
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

            # Prevent duplicate entries for same day
            if not ((df["Name"] == name) & (df["Date"] == date)).any():
                df = pd.concat([df, pd.DataFrame([[name, date, time]], columns=df.columns)], ignore_index=True)
                df.to_csv(attendance_file, index=False)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Face Recognition Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
