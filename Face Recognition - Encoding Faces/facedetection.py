from deepface import DeepFace
import cv2 as cv
import numpy as np
from pathlib import Path

# DeepFace.build_model("Facenet")

proto_text = "Face Recognition - Encoding Faces/deploy.prototxt"
model = "Face Recognition - Encoding Faces/res10_300x300_ssd_iter_140000.caffemodel"

net = cv.dnn.readNetFromCaffe(proto_text, model)

text = ""
Video = cv.VideoCapture(0)
running = True
path_of_image = Path("Face Recognition - Encoding Faces/images/")
check = 0
continue_varification = True
verified = False

with open("Face Recognition - Encoding Faces/save.txt", "r") as img_ind:
    saved_index = int(img_ind.read().strip())




while running:
    ret, frame = Video.read()
    frame = cv.flip(frame, 1)
    h,w = frame.shape[:2]

    if not ret:
        break

    blob = cv.dnn.blobFromImage(image=cv.resize(frame, (300,300)),scalefactor=1, size=(300,300),mean=(104, 177, 123))

    net.setInput(blob)

    face_detection = net.forward()
    cv.putText(frame, text, (70, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)

    for i in range(face_detection.shape[2]):
        confidence = face_detection[0,0,i,2]

        if confidence > 0.7:
            check += 1

            box = face_detection[0,0,i,3:7] * np.array([w,h,w,h])
            x1,y1,x2, y2 = box.astype('int')

            face_cropping = frame[y1:y2, x1:x2]

            cv.rectangle(frame,(x1,y1),(x2,y2),(255,64,64), 3)

            if text == "r":
                continue_varification = True
                verified = False

            if text == "verify":

                if continue_varification:

                    for img_file in path_of_image.glob("*.jpg"):
                        face = DeepFace.verify(img1_path=face_cropping, 
                                            img2_path=str(img_file),
                                            model_name="Facenet",
                                            enforce_detection=False)
                
                        if face["verified"] == True:
                            verified = True
                            continue_varification = False
                            break

                    if face["verified"] == False:
                        continue_varification = False

                
                if verified:
                    cv.putText(frame, "You are Verified To Enter", (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,), 3)
                elif not verified:
                    cv.putText(frame, "You are NOT Verified To Enter", (x1-20, y1-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)


                    

                    

    cv.imshow("Video", frame)

    key = cv.waitKey(25) & 0xFF

    if key & 0xFF == ord('x'):
        running = False
        
    elif key == ord('\r'):  # Enter to finish
        if text == "save":
            cv.imwrite(f"Face Recognition - Encoding Faces/images/{saved_index:03d}.jpg", face_cropping)
            print(f"Captured Image: {saved_index:03d}")

            saved_index += 1

            with open("Face Recognition - Encoding Faces/save.txt", "w") as f:
                f.write(str(saved_index))


            
        
    elif key == 8:  # Backspace
        text = text[:-1]
    elif key == ord('c'):
        text = ""
    elif 32 <= key <= 126:  # Printable ASCII
        text += chr(key)
    

Video.release()
cv.destroyAllWindows()