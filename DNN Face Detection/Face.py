import cv2 as cv
import numpy as np

##load up the pretrained DNN model
proto_text = "DNN Face Detection/deploy.prototxt"
model = "DNN Face Detection/res10_300x300_ssd_iter_140000.caffemodel"

DNN = cv.dnn.readNetFromCaffe(proto_text, model)


#video loadup
capture = cv.VideoCapture(0)
running = True

while running:

    #read the video
    ret, video = capture.read()
    video = cv.flip(video, 1)
    h, w = video.shape[:2]


    #making food for the DNN that it likes to take in
    blob = cv.dnn.blobFromImage(image= cv.resize(video, (300,300)),scalefactor= 1, size=(300,300), mean=(104, 177, 123))

    #feeding the image to the DNN
    DNN.setInput(blob)


    #runs the DNN to get the output
    detection = DNN.forward()

    if not ret:
        break

    
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        
        if confidence > 0.7:
            box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv.rectangle(video, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv.imshow('video', video)

    if cv.waitKey(25) & 0xFF == ord('x'):
        running = False

capture.release()
cv.destroyAllWindows()

