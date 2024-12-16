# **Driver Drowsiness Detection System**

## **Overview**

This project implements a **Driver Drowsiness Detection System** using computer vision, deep learning, and facial landmark detection techniques. The system continuously monitors the driver's face and eyes to detect signs of drowsiness, such as eye closure and yawning. When drowsiness is detected, an alarm is sounded to alert the driver.

---

## **Features**

* **Real-time video stream processing**: Captures live video from the webcam.  
* **Face and eye detection**: Uses OpenCV's Haar Cascade Classifiers for detecting faces and eyes.  
* **Deep learning for eye state classification**: A Keras deep learning model predicts if the eyes are open or closed.  
* **Yawn detection**: Uses dlib's facial landmarks to detect yawning.  
* **Audio alerts**: Plays an alarm sound to notify the driver of drowsiness.

---

## **Project Structure**

├── ddd.ipynb                 \# Jupyter Notebook for system development and testing  
├── model\_best.h5             \# Trained deep learning model for eye state classification  
├── pls deploy.py             \# Main Python script to run the Drowsiness Detection System  
├── shape\_predictor\_68\_face\_landmarks.dat  \# Pre-trained dlib shape predictor for facial landmarks  
---

## **Prerequisites**

### **Libraries**

The following Python libraries are required to run the project:

* `OpenCV`  
* `NumPy`  
* `SciPy`  
* `dlib`  
* `keras`  
* `imutils`  
* `playsound`

Install the required libraries using the following command:

pip install opencv-python-headless numpy scipy dlib keras imutils playsound

### **Files**

Make sure the following files are available in the same directory as the main script:

* `model_best.h5`: The trained Keras model for eye detection.  
* `shape_predictor_68_face_landmarks.dat`: The dlib shape predictor file for facial landmark detection.  
* Haar Cascade files for face and eye detection.

---

## **Usage**

1. **Run the main script**:  
   python pls deploy.py  
2. The system will access the webcam and start the video stream.  
3. The system will display the video frame with detected face, eyes, and status messages like "Eyes Closed" or "Drowsiness Alert".  
4. If the driver closes their eyes for too long or yawns, an alert sound will be played.  
5. Press `q` to quit the application.

---

## **How It Works**

1. **Face Detection**:  
   * Uses OpenCV Haar Cascades to detect faces in the video stream.  
2. **Eye Detection**:  
   * Detects eyes within the detected face region using the Haar Cascade method.  
   * The detected eyes are processed by a deep learning model to classify if they are open or closed.  
3. **Drowsiness Detection**:  
   * If the driver's eyes remain closed for more than a set threshold of consecutive frames, a drowsiness alert is triggered.  
   * Yawning is detected using dlib's 68-point facial landmarks and calculating the distance between the upper and lower lip.  
4. **Alarm System**:  
   * An alarm sound is played to alert the driver when drowsiness is detected.

---

## **Configuration**

* **EYE\_AR\_THRESH**: Eye aspect ratio threshold to classify eyes as open or closed.  
* **EYE\_AR\_CONSEC\_FRAMES**: Number of consecutive frames to consider eyes as closed.  
* **YAWN\_THRESH**: Lip distance threshold to classify a yawn.

You can modify these constants in `pls deploy.py` to suit your preferences.

---

## **Troubleshooting**

* **Webcam Not Detected**: Ensure the webcam is properly connected and accessible by OpenCV.  
* **Missing Files**: Ensure `model_best.h5` and `shape_predictor_68_face_landmarks.dat` are in the working directory.  
* **Audio Not Playing**: Check if the file path for the alarm sound in `pls deploy.py` is correct.

---

## **Acknowledgments**

* [OpenCV](https://opencv.org/) for face and eye detection.  
* [dlib](http://dlib.net/) for facial landmark detection.  
* [Keras](https://keras.io/) for deep learning.