import cv2 as cv
import numpy as np
from pathlib import Path

image_dir = Path("Smart Image Analyzer/images")

for image_path in image_dir.iterdir():
    img = cv.imread(image_path)
    

    #image resolution
    H, W = img.shape[:2]

    image_resoultion = (f"{H} X {W}")

    #image histogram

    blue_hist = cv.calcHist([img],[0],None,[256],[0,256])
    green_hist = cv.calcHist([img],[1],None,[256],[0,256])
    red_hist = cv.calcHist([img],[2],None,[256],[0,256])

    total_blue = blue_hist.sum()
    total_green = green_hist.sum()
    total_red = red_hist.sum()

    #image brightness and contrass
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    brightness = np.mean(gray_img)
    contras = np.std(gray_img)
    

    #blurrynesss detection

    lap = cv.Laplacian(gray_img,cv.CV_64F)
    lap_var = lap.var()

    
    #edge detection
    edge = cv.Canny(gray_img, threshold1=50,threshold2=150)
    cv.imshow("Edge",edge)

    cv.waitKey(0)
    cv.destroyAllWindows()

    print(f"\n{'-'*50}")
    print(f"ANALYZING: {image_path.name}")
    print(f"\n{'-'*50}")

    if lap_var < 100:
        print("it is blurry")
    elif lap_var > 100:
        print("It is not blurry")


    print(f"the Laplacian value is {lap_var}")



    print(f"The brightness is: {brightness:.2f} and the contrass is: {contras:.2f}")

    print(f"The total Color Histogram is:\n Red: {total_red}\n Green: {total_green}\n Blue: {total_blue}")

    print(f"Image resolution = {image_resoultion}")

    print(f"Image file size is {img.size}")