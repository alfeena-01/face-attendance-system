import cv2
import os

# Create dataset folder if not exists
dataset_path = "data"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Initialize webcam
cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

user_id = input("Enter User ID or Name: ")
count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        # Save face image
        cv2.imwrite(f"{dataset_path}/{user_id}_{count}.jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Capturing Faces", frame)

    # Break after 50 samples
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
        break

cap.release()
cv2.destroyAllWindows()
print("Dataset collection complete!")
