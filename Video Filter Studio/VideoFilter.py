import cv2 as cv

Video = cv.VideoCapture(0)
running = True
save_index = 0
def normal(frame):
    return frame
def gray_conv(frame):
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

def blur_vid(frame):
    return cv.GaussianBlur(frame,(111,111), 0)

def edge_detection(frame):
    return cv.Canny(frame,threshold1=50,threshold2=150)

def color_inverse(frame):
    return cv.bitwise_not(frame)


filter = {
    "normal":normal,
    "gray":gray_conv,
    "blur": blur_vid,
    "canny": edge_detection,
    "inverse": color_inverse,

}


mode = "normal"

while running:
    isTrue, Frame = Video.read()
    key = cv.waitKey(1)
    
    
    ##resets it all back to normal
    if key == ord('r'):
        mode = "normal"
    ##convert video to grayscale
    elif key == ord('1'):
        mode = "gray"
    elif key == ord('2'):
        mode = "blur"
    elif key == ord('3'):
        mode = "canny"
    elif key == ord('4'):
        mode = "inverse"
    
    Frame = cv.flip(Frame,1)
    output = filter[mode](Frame)

    if key == ord('s'):
        cv.imwrite(f"Video Filter Studio/Save_img/{save_index:03d}.jpg", output)
        save_index += 1

    cv.imshow("Video", output)
    

    if key & 0xFF==ord('x'):
        running = False

Video.release()
cv.destroyAllWindows()