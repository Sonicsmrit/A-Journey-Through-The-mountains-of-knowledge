import cv2 as cv

face_cascade = cv.CascadeClassifier('FaceDetector/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('FaceDetector/haarcascade_eye_tree_eyeglasses.xml')

Video = cv.VideoCapture(0)

while True:
    ret, vid = Video.read()

    vid = cv.flip(vid, 1)

    if not ret:
        break
    
    gray = cv.cvtColor(vid, cv.COLOR_BGR2GRAY)

    face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
    eye = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    faces = len(face)
    
    text = f"Number of Faces {faces}"

    for (x, y, w, h) in face:
        cv.rectangle(vid, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv.putText(vid, text,(0,30), cv.FONT_HERSHEY_COMPLEX,1,(0, 255, 0), 3)
    
    for (x,y,w,h) in eye:
        cv.circle(vid, (x+w//2,y+h//2), 30, (0,0,0),3)
        
    




    cv.imshow("Video", vid)

    if cv.waitKey(25) & 0xFF== ord('x'):
        break


Video.release()
cv.destroyAllWindows()