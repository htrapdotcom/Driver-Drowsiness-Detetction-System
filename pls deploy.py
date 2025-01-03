import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import dlib
import keras
from keras.utils import img_to_array
from playsound import playsound
import os
import imutils
import time

# Function to start the alarm sound
def start_alarm(sound):
    playsound(sound)

# Initialize the alarm sound file path
alarm_sound = "D:\DDD\iphone-alarm-vs-android-alarm-128-ytshorts.savetube.me.mp3"

# Initialize the OpenCV face and eye cascade classifiers
face_cascade = cv2.CascadeClassifier("D:\DDD\DL_Driver-drowsiness-detection-main\DL_Driver-drowsiness-detection-main\haarcascade\haarcascade_frontalface_default.xml")
left_eye_cascade = cv2.CascadeClassifier("D:\DDD\DL_Driver-drowsiness-detection-main\DL_Driver-drowsiness-detection-main\haarcascade\haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier("D:\DDD\DL_Driver-drowsiness-detection-main\DL_Driver-drowsiness-detection-main\haarcascade\haarcascade_righteye_2splits.xml")

# Initialize the deep learning model for eye drowsiness detection
model = keras.models.load_model("D:\DDD\DL_Driver-drowsiness-detection-main\DL_Driver-drowsiness-detection-main\model_best.h5")

# Initialize dlib's face detector and shape predictor for facial landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:\DDD\shape_predictor_68_face_landmarks.dat")

# Function to calculate lip distance
def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

# Initialize constants and variables for drowsiness detection
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0

# Initialize the video stream
vs = VideoStream(src=0).start()
time.sleep(1.0)

# Main loop for drowsiness detection
while True:
    # Read a frame from the video stream
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using OpenCV's cascade classifier
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Loop over the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes using OpenCV's cascade classifiers
        left_eye = left_eye_cascade.detectMultiScale(roi_gray)
        right_eye = right_eye_cascade.detectMultiScale(roi_gray)
        
        for (x1, y1, w1, h1) in left_eye:
            cv2.rectangle(roi_color, (x1, y1),
                          (x1 + w1, y1 + h1), (0, 255, 0), 1)
            eye1 = roi_color[y1:y1+h1, x1:x1+w1]
            eye1 = cv2.resize(eye1, (145, 145))
            eye1 = eye1.astype('float') / 255.0
            eye1 = img_to_array(eye1)
            eye1 = np.expand_dims(eye1, axis=0)
            pred1 = model.predict(eye1)
            status1 = np.argmax(pred1)
            break

        for (x2, y2, w2, h2) in right_eye:
            cv2.rectangle(roi_color, (x2, y2),
                          (x2 + w2, y2 + h2), (0, 255, 0), 1)
            eye2 = roi_color[y2:y2 + h2, x2:x2 + w2]
            eye2 = cv2.resize(eye2, (145, 145))
            eye2 = eye2.astype('float') / 255.0
            eye2 = img_to_array(eye2)
            eye2 = np.expand_dims(eye2, axis=0)
            pred2 = model.predict(eye2)
            status2 = np.argmax(pred2)
            break

        # If the eyes are closed, start counting
        if status1 == 2 and status2 == 2:
            COUNTER += 1
            cv2.putText(frame, "Eyes Closed, Frame count: " + str(COUNTER),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            # If eyes are closed for 10 consecutive frames, start the alarm
            if COUNTER >= 10:
                cv2.putText(frame, "Drowsiness Alert!!!", (100, frame.shape[0]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if not alarm_status:
                    alarm_status = True
                    t = Thread(target=start_alarm, args=(alarm_sound,))
                    t.daemon = True
                    t.start()
        else:
            cv2.putText(frame, "Eyes Open", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            COUNTER = 0
            alarm_status = False



    # Detect facial landmarks using dlib
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        distance = lip_distance(shape)

        if distance > YAWN_THRESH:
            cv2.putText(frame, "Yawn Alert", (100, frame.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if alarm_status2 == False and saying == False:
                alarm_status2 = True
                t = Thread(target=start_alarm, args=('take some fresh air sir',))
                t.daemon = True
                t.start()
        else:
            alarm_status2 = False

    # Display the frame
    cv2.imshow("Drowsiness Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
vs.stop()
