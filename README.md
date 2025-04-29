# Driver Drowsiness Detection System

## Description

This project provides a system for detecting driver drowsiness in real-time. It uses a webcam to monitor the driver's eyes and mouth, analyzing eye closure and yawning patterns. If drowsiness is detected, an alarm is triggered to alert the driver.

The project consists of two main parts:

- `ddd.ipynb`: A Jupyter Notebook containing the code for training the drowsiness detection model. This includes data loading, model architecture, training, and evaluation.
- `pls deploy.py`: A Python script for deploying the drowsiness detection system in real-time. This script captures video from a webcam, processes the frames to detect drowsiness, and triggers an alarm if necessary.

---

## Features

- **Real-time Drowsiness Detection:** Monitors eye closure and yawning in real-time using a webcam.
- **Eye Closure Detection:** Calculates the Eye Aspect Ratio (EAR) to detect prolonged eye closure.
- **Yawn Detection:** Measures lip distance to detect yawning.
- **Alarm System:** Triggers an audio alarm to alert the driver.
- **Haar Cascade Classifiers:** Uses Haar cascade classifiers for face and eye detection.
- **Dlib Facial Landmarks:** Employs dlib's facial landmark predictor for precise facial feature tracking.
- **Keras Model:** Utilizes a Keras-based deep learning model for eye state classification.

---

## Requirements

- Python 3.6 or higher
- OpenCV (`cv2`)
- NumPy (`numpy`)
- Pandas (`pandas`) (for `ddd.ipynb`)
- Matplotlib (`matplotlib`) (for `ddd.ipynb`)
- SciPy (`scipy`)
- imutils
- dlib
- Keras
- playsound

Install the required Python libraries using pip:

```bash
pip install opencv-python numpy pandas matplotlib scipy imutils dlib keras playsound
```

(Note: You might need to install TensorFlow separately if Keras doesn't automatically install it.)

---

## Setup

### Clone the Repository

```bash
git clone [repository_url]
cd [repository_directory]
```

### Install Dependencies

```bash
pip install -r requirements.txt  # If you create a requirements.txt
# or install them manually as listed in the "Requirements" section.
```

### Download Necessary Files

- Download the Haar cascade XML files (`haarcascade_frontalface_default.xml`, `haarcascade_lefteye_2splits.xml`, `haarcascade_righteye_2splits.xml`) and place them in the `haarcascade` directory. These files are typically available in the OpenCV repository.
- Download the dlib facial landmark predictor (`shape_predictor_68_face_landmarks.dat`) and place it in the appropriate directory. You can obtain this file from the dlib website or repository.
- Ensure the drowsiness detection model (`DL_Driver-drowsiness-detection-main...`) is correctly placed.
- Place an alarm sound file (`iphone-alarm-vs-android-alarm-128-ytshorts.savetube.me.mp3` or your preferred sound) in the correct location.

> ⚠️ Adjust the file paths in the code to match your directory structure.

---

## Usage

### Training the Model

1. Open and run the `ddd.ipynb` Jupyter Notebook.
2. Make sure to adjust the data paths within the notebook to point to your training data.

### Running Real-time Detection

1. Ensure your webcam is connected.
2. Open a terminal or command prompt.
3. Navigate to the project directory.
4. Run the `pls deploy.py` script:

```bash
python pls deploy.py
```

The script will capture video from your webcam and display the output. If drowsiness is detected, an alarm will sound.

---

## Code Explanation

### `ddd.ipynb`

- Loads and preprocesses the drowsiness dataset.
- Defines and trains a convolutional neural network (CNN) model using Keras.
- Evaluates the model's performance.

### `pls deploy.py`

- Captures video from a webcam.
- Detects faces and eyes using Haar cascade classifiers.
- Uses dlib's facial landmark predictor to get precise eye and mouth coordinates.
- Calculates the Eye Aspect Ratio (EAR) to detect eye closure.
- Calculates lip distance to detect yawning.
- Uses the trained Keras model to classify eye states.
- Triggers an alarm if drowsiness is detected (prolonged eye closure or frequent yawning).

---

## Important Notes

- **File Paths:** The code relies on specific file paths. Modify these paths to match your local setup.
- **Environment:** It's highly recommended to use a virtual environment to manage project dependencies.
- **Model Training:** Training the model in `ddd.ipynb` can be computationally intensive and may require a GPU.
- **Accuracy:** The system's accuracy depends on training data quality, model architecture, and environmental conditions (e.g., lighting).
- **Calibration:** Adjust thresholds (e.g., EAR, yawn) for optimal performance.

---

## Future Enhancements

- Improve model accuracy with more data and advanced architectures.
- Implement more robust face and eye tracking.
- Add features like head pose estimation.
- Optimize performance for real-time processing.
- Integrate with vehicle systems.

---

## Author

Parth

---

## License

[MIT License](https://opensource.org/licenses/MIT) or any other license you prefer.

