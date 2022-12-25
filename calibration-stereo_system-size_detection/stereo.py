# %matplotlib inline
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt 
import numpy as np
from glob import glob 

all_right_corners=[]
all_left_corners=[]
all_3d_points=[]

idx = [1,3,6]
idx = [1,2,3,4,5,6,7,8,9,10]
valid_idxs = []

for i in idx:
    im_left = cv2.imread("stereo_img/left%02d.jpg"%i)
    im_right = cv2.imread("stereo_img/right%02d.jpg"%i)

    ret_left,left_corners = cv2.findChessboardCorners(im_left,(9,6))
    ret_right,right_corners = cv2.findChessboardCorners(im_right,(9,6))

    if ret_left and ret_right:
        valid_idxs.append(i)
        all_right_corners.append(right_corners)
        all_left_corners.append(left_corners)
        all_3d_points.append(world_points)


retval, _, _, _, _, R, T, E, F=cv2.stereoCalibrate(all_3d_points, all_left_corners, all_right_corners, mtx,dist,mtx,dist,(im.shape[1],im.shape[0]), flags=cv2.CALIB_FIX_INTRINSIC)
print(retval) 


pts1 = np.int32(left_corners)
pts2 = np.int32(right_corners)
F2, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)

print(F)
print(F2)

selected_image = 2
left_im=cv2.imread("left%02d.jpg"&valid_idxs[selected_image])
right_im=cv2.imread("right%02d.jpg"&valid_idxs[selected_image])
left_corners=all_left_corners[selected_image].reshape(-1,2)
right_corners=all_right_corners[selected_image].reshape(-1,2)

plt.figure(figsize=(16,8))
plt.subplot(121)
plt.imshow(left_im)
plt.subplot(122)
plt.imshow(right_im)
plt.show()

cv2.circle(left_im,(left_corners[0,0],left_corners[0,1]),10,(0,0,255),10)
cv2.circle(right_im,(right_corners[0,0],right_corners[0,1]),10,(0,0,255),10)

plt.figure(figsize=(16,8))
plt.subplot(121)
plt.imshow(left_im[...,::-1])
plt.subplot(122)
plt.imshow(right_im[...,::-1])
plt.show()


lines_right = cv2.computeCorrespondEpilines(all_left_corners[selected_image], 1,F)
lines_right_F2 = cv2.computeCorrespondEpilines(all_left_corners[selected_image], 1,F2)

print(lines_right.shape)
lines_right = lines_right.reshape(-1,3)
print(lines_right.shape)
lines_right_F2 = lines_right_F2.reshape(-1,3)


def drawLine(line,image,line_color):
    a = line[0]
    b = line[1]
    c = line[2]

    x0,y0 = map(int,[0, -c/b])
    x1,y1 = map(int, [image.shape[1], -(c+a*image.shape[1])/b])

    cv2.line(image,(x0,int(y0)),(x1,int(y1)),line_color,3)


drawLine(lines_right[0],right_im,(0,255,255))

print(right_im.shape[1])
plt.figure(figsize=(16,8))
plt.subplot(121)
plt.imshow(left_im[...,::-1])
plt.subplot(122)
plt.imshow(right_im[...,::-1])
plt.show()



R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx, dist, mtx, dist, (left_im.shape[1],left_im.shape[0]), R, T)

