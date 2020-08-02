import numpy as np
import cv2
import imutils #basic image processing functions (translation,rotation,resizing)
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

#Read the original image
original = cv2.imread('CarImages/car5.jpg')

#Resize the original image for displaying on the screen
#Parameter: width=500, the size the image is displayed with
original = imutils.resize(original,width=500)

#Display the original image
cv2.imshow("Original",original)
cv2.waitKey(0)

#Convert the image to gray scale
#Reference: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
gray_img = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Scale Image", gray_img)
cv2.waitKey(0)

#Bilateral filter image
#Bilateral filter removes noise while keeping edges sharp
#Ref: https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html
bilateral_img = cv2.bilateralFilter(gray_img,11,17,17)
#Parameters: bilateralFilter(src,dst,d,sigmaColor)
cv2.imshow("Bilateral Filter",bilateral_img)
cv2.waitKey(0)

#Canny Edge Detection
#Ref: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
#cv2.Canny(src,minVal,maxVal)
canny_img = cv2.Canny(bilateral_img,80,200)
cv2.imshow("Canny Edge Detection",canny_img)
cv2.waitKey(0)

#Obtain contours on the Canny edged image
#Ref: https://docs.opencv.org/trunk/d3/dc0/group__imgproc__shape.html#gga819779b9857cc2f8601e6526a3a5bc71a48b9c2cb1056f775ae50bb68288b875e
#cv2.findContours(src, mode, method)
cnts, placeholder = cv2.findContours(canny_img.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

#Draw found contours on the original image
#Ref: https://docs.opencv.org/trunk/d6/d6e/group__imgproc__draw.html#ga746c0625f1781f1ffc9056259103edbc
cnt_original = original.copy()
cv2.drawContours(cnt_original,cnts,-1,[0,255,0],2)
cv2.imshow("All Contoured Original",cnt_original)
cv2.waitKey(0)

#Sort Contours based on are
cnts = sorted(cnts,key=cv2.contourArea,reverse=True)[:30]
PlateCnt = None #plate holder for the contour of the license plate

#Draw the top specified contours
top_cnt_original = original.copy()
cv2.drawContours(top_cnt_original,cnts,-1,[0,255,0],2)
cv2.imshow("Top Contoured Original",top_cnt_original)
cv2.waitKey(0)

#Loop over the contours to find 4
for c in cnts:
    perimeter = cv2.arcLength(c,True) #closed = True
    approx_Curve = cv2.approxPolyDP(c,0.02*perimeter,True) #closed = True
    if len(approx_Curve)==4: # if the polygon has 4 corners
        PlateCnt = approx_Curve #PlateCnt is the contour of the plate
        x,y,w,h = cv2.boundingRect(c)
        plate_img = original[y:y+h, x:x+w]
        cv2.imwrite('LicensePlateImages/license.png',plate_img)
        break
#Draw contour around the license plate in the original image
cv2.drawContours(original,[PlateCnt],-1,[0,255,0],2)
cv2.imshow("Plate on Original",original)
cv2.waitKey(0)
#Show the cropped plate image
cropped_plate = cv2.imread('LicensePlateImages/license.png')
cv2.imshow("Cropped Plate",cropped_plate)
cv2.waitKey(0)

#Perform erosion
kernel = np.ones((2,2),np.uint8)
eroded_cropped_plate = cv2.erode(cropped_plate.copy(),kernel,iterations=1)
cv2.imshow("Eroded Cropped Plate",eroded_cropped_plate)
cv2.waitKey(0)

#Thresholding
ret, threshold_plate = cv2.threshold(eroded_cropped_plate,127,255,cv2.THRESH_BINARY)
cv2.imshow("Threshold",threshold_plate)
cv2.waitKey(0)

plate_num = pytesseract.image_to_string(threshold_plate,lang='eng')
print("The license plate number is: ",plate_num)





























