import cv2 as cv

Video = cv.VideoCapture(0)
running = True

kernal = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
fgbg = cv.createBackgroundSubtractorMOG2()



while running:
    ret, frame = Video.read()

    frame = cv.flip(frame,1)


    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


    fgmask = fgbg.apply(frame_gray)
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernal)
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernal)


    contors, hierarcy = cv.findContours(fgmask,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)



    for cnt in contors:

        if cv.contourArea(cnt)<200:
            continue

        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv.putText(frame, "Motion Detected", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



    cv.imshow("Web Cam", frame)
    cv.imshow("mask", fgmask)

    if cv.waitKey(25) == ord('x'):
        running = False


Video.release()
cv.destroyAllWindows()