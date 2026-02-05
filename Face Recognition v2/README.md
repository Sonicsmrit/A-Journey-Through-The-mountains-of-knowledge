# Real-time face recognition using pre-computed embeddings. 20x faster than the previous attempt.

## Requirements

```bash
pip install deepface opencv-python numpy
```
## Files
- build_database.py - Database management
- recognize_faces.py - Main recognition script
- face_database.pkl - Stored embeddings
- index.txt - Counter

## Run
```bash
python recognize_faces.py
```

## Controls
- Type `o123` + ENTER - Save face
- **X** - Exit

## How It Works
> Startup: Load pre-computed embeddings
> Each frame: Detect face → Extract embedding (every 20 frames) → Compare to database → Display match result

## Parameters

- Recognition threshold: 6.0 (Euclidean distance)
- Frame skip: 20 (process every 20th frame)
- Model: Facenet (128D embeddings)