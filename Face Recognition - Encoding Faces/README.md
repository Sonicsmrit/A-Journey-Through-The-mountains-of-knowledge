# Face Verification System

Real-time face detection and verification using OpenCV DNN and DeepFace.

## Features

> Detects faces using OpenCV DNN (SSD with ResNet-10 backbone).

> Save authorized faces for future verification.

> Manual real-time verification against saved images using Facenet embeddings.

# Model Used

> Face Detection: SSD (Single Shot Detector)

> Backbone: ResNet-10

> Framework: Caffe

> Input size: 300×300

> Face Verification: DeepFace (Facenet)

# Requirements

- Python 3.11+

- OpenCV

- NumPy

- DeepFace

# Install dependencies:

```bash
pip install opencv-python numpy deepface
```

# Usage

Run the script and use keyboard inputs:

- Type "save" → Enter to save the current face.

- Type "varify" → Enter to verify the current face.

- Type "c" → Clear typed text.

-Type "r" → Resets the the verification process so it can detect new faces to verify.

- Press x → exit the program.

