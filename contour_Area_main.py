import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# read image
#img = cv.imread('contour_2.png')
img = cv.imread('contour1.png')
cv.imshow("original image", img)
cv.waitKey(0)


# transfer to grey-scale image
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("gray image", img_gray)
cv.waitKey(0)


# Binary threshold
#gray_thresh = 200
gray_thresh = 150
#gray_thresh = 30

ret,thresh = cv.threshold(img_gray,gray_thresh,255,0)
cv.imshow("Binary mage", thresh)
cv.waitKey(0)


# find countours
contours, hierarchy = cv.findContours(image=thresh, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
image_copy = img.copy()
#cv.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 255 ), thickness=5,lineType=cv.LINE_AA)
cv.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(55, 55, 155), thickness=2)
cv.imshow('None approximation', image_copy)
cv.waitKey(0)
cv.destroyAllWindows()


# export the binary-shreshold image and contoured image
file_output_binary = "contour_2_output_binary.png"
cv.imwrite(file_output_binary, thresh)
file_output_contour = "contour_2_output_contour.png"
cv.imwrite(file_output_contour, image_copy)


# Calculate the area of all the contours
contours_num = len(contours)
print(" The number of contours = " + str(contours_num))
area = []
contours_area = np.double(0.0)
for i in range(contours_num):
    area.append(cv.contourArea(contours[i]))
    contours_area = contours_area + area[i]
#print(len(area))
#area_total_sum = sum(area)
print("The total contoured (white) area = " + str(contours_area))

# Calculate the area of the image:
img_dimensions = img.shape
#print("the dimensions of the image = " + str(img_dimensions))
img_area = np.double(img_dimensions[0] * img_dimensions[1])
print("the total area of the image = " + str(img_area))

contours_area_ratio = round(contours_area/img_area, 2)
contours_non_area_ratio = round(1-contours_area/img_area, 2)
print("the ratio between contoured (white) area and total area = " + str(100*contours_area_ratio) + "%")
print("the ratio between contoured (black) area and total area = " + str(100*contours_non_area_ratio) + "%")



# plot the images
plt.subplot(121),plt.imshow(img)
plt.xlabel('x-pixels')
plt.ylabel('y-pixels')
plt.title('The original image')
plt.subplot(122),plt.imshow(image_copy)
plt.xlabel('x-pixels')
plt.ylabel('y-pixels')
plt.title('The image with contour')
plt.show()

