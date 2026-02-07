# Face Recognition Attendance System

> A real-time attendance system with face recognition using OpenCV for face detection and DeepFace (FaceNet) for face recognition.

## Features

- Real-time face detection (OpenCV DNN)

- Face recognition using FaceNet embeddings

- User registration with password

- Automatic attendance recording

- Webcam-based system

## Requirements
```bash
pip install opencv-python deepface numpy
```

## Run
```bash
python attendence.py
```

## Controls
### Key	Action
- p	(Enter password mode)
- Enter	(Confirm input)
- Backspace	(Delete character)
-x	(Exit)

## Password for password mode
> "o123"

## How It Works

- Capture webcam frame

- Detect face using OpenCV DNN

- Generate FaceNet embedding

- Compare with stored embeddings

- Mark attendance if match found

## Notes

> Recognition runs every 20 frames for performance

> Matching uses Euclidean distance

> Threshold value: 6

> Good lighting improves accuracy

## Files

- atttendence.py – main program

- database.py – face data & attendance handling

- DNN/ – face detection model files

- Data/ - all the database files