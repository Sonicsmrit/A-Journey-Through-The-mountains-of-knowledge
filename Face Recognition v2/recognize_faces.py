from deepface import DeepFace
import cv2 as cv
import numpy as np
import build_database

proto_text = "Face Recognition v2/DNN/deploy.prototxt"
model = "Face Recognition v2/DNN/res10_300x300_ssd_iter_140000.caffemodel"


DNN = cv.dnn.readNetFromCaffe(proto_text, model) ##DNN model initialization

data_dict = build_database.load_database() ##loads the database in this file

vid = cv.VideoCapture(0)
running = True
text = ""
capture = 0
last_difference = float("inf")

while running:
    ret, frame = vid.read()
    frame = cv.flip(frame, 1)

    h,w = frame.shape[:2]
    key = cv.waitKey(25)

    
    if not ret:
        break
    

    blob = cv.dnn.blobFromImage(
        frame, scalefactor=1, size=(300,300), mean=(104,177,123)
    )

    DNN.setInput(blob=blob)

    face_detect = DNN.forward() ##face detection forwarded to DNN

    cv.putText(frame, text, (30,40),cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3) #text on screen

    for i in range(face_detect.shape[2]): ##face detection loop

        confidence = face_detect[0,0,i,2] ##confidence of it being a face

        if confidence > 0.7: ##if greater than 70%

            face_box = face_detect[0,0,i,3:7] * np.array([w,h,w,h])

            x1, y1, x2, y2 = face_box.astype("int")

            face_crop = frame[y1:y2, x1:x2]

            cv.rectangle(frame,(x1,y1), (x2,y2),(255,0,0),3)

            ##values
            smallest_distance = float('inf')
            Threshold_value = 6

            ##smoother processing by only processing 20th frame each time
            capture +=1
            if capture%20 == 0:
                #get the face data of the face on screen
                current_embedding = DeepFace.represent(
                    img_path=face_crop,
                    model_name="Facenet",
                    enforce_detection=False
                )
                
                ##loop to compare all the data from the database and the current face data
                for name, stored_embedding in data_dict.items():
                    #finding the difference between the current face and the face in the database
                    difference = np.linalg.norm(np.array(current_embedding[0]["embedding"]) - np.array(stored_embedding))

                    #putting the smallest difference in the smallest_distance to see if there is a match
                    if difference < smallest_distance:
                        smallest_distance = difference

                        last_difference = smallest_distance ##keeping it in the last_diff variable so the data doesnt flicker
            
            if last_difference < Threshold_value: ##comparing with the threshold to see if the value is a match or not
                cv.rectangle(frame,(x1,y1), (x2,y2),(0,255,0),3)
                cv.putText(frame, "Valid User",(x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,), 3)
            else:
                cv.rectangle(frame,(x1,y1), (x2,y2),(0,0,255),3)
                cv.putText(frame, "In Valid User",(x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)






    ##showing on screen
    cv.imshow("Webcam", frame)


    ##taking text input
    if key & 0xFF == ord('x'):
        break

    elif key == ord("\r"):
        if text == "o123":  ##password to save your image
            data_dict = build_database.database(face_crop)

    elif key == 8:  # Backspace
        text = text[:-1]
    elif key == ord('c'):
        text = ""
    elif 32 <= key <= 126:  # Printable ASCII
        text += chr(key)