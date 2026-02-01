Real-time face detection using Haar Cascade classifier. Detects multiple faces simultaneously from webcam feed.
 Requirements
bashpip install opencv-python

 Files Needed

haarcascade_frontalface_default.xml (OpenCV pre-trained model)

How to Run
bashpython face_detector.py
Controls

X - Exit program

How It Works

Loads pre-trained Haar Cascade face detection model
Captures webcam frames
Converts to grayscale for processing
Detects faces using detectMultiScale()
Draws bounding boxes around detected faces
Displays face count on screen

Key Parameters

scaleFactor=1.1 - Image pyramid scaling (lower = slower but more accurate)
minNeighbors=10 - Detection quality threshold (higher = fewer false positives)

Features

✓ Real-time detection
✓ Multiple face support
✓ Face counting
✓ Mirror flip for natural viewing

🚨 Limitations

Works best with frontal faces
Struggles with profile views
Affected by lighting conditions


🎓 Learning Outcomes

Using pre-trained ML models
Cascade classifier concept
Real-time AI inference
Parameter tuning for accuracy vs performance