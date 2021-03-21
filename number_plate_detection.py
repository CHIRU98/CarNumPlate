import cv2
import numpy as np 
import matplotlib.pyplot as plt
import imutils

#read the image
img = cv2.imread(r"G:\Open Cv\Car Number Plate Detection\image1.jpg") # BGR
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #coverted into gray to easily can remove the noise
plt.figure(figsize=(15,15))
# plt.imshow(gray,cmap='gray')
# plt.show()


#apply filter to remove the noise in the image
aply_filter = cv2.bilateralFilter(gray,11,17,17)

#edge detection
edg_det = cv2.Canny(aply_filter,30,200)
# plt.imshow(edg_det,cmap="gray")
# plt.show()


#find the countours in the image
keypoints = cv2.findContours(edg_det.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)

contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = []
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break


mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0,255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

# plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
# plt.show()


(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]

plt.imshow(cropped_image,cmap="gray")
plt.show()




