import cv2 as cv

img = cv.imread("Advance Image Processing/image/Squidbird.jpg")


#pipeline

#smoothen image
bilateral_filter = cv.bilateralFilter(img, d=5,sigmaColor=120,sigmaSpace= 120)


#enhance image detail
lab = cv.cvtColor(bilateral_filter, cv.COLOR_BGR2LAB)

l,a,b = cv.split(lab)

clahe = cv.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))

l = clahe.apply(l)

enhanced = cv.merge((l, a, b))
enhanced = cv.cvtColor(enhanced, cv.COLOR_LAB2BGR)


##sharpness of image
blurred = cv.GaussianBlur(enhanced, (0, 0), 3)
sharpened = cv.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

#show


cv.imshow("Original", img)
cv.imshow("Enhanced", sharpened)


cv.waitKey(0)
cv.destroyAllWindows()