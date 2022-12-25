# coding=utf-8
# %matplotlib inline
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt 
import numpy as np
from glob import glob 
# corners=corners.reshape(-1,2) 
# im_right_vis = im.copy() # copy by value 
# cv2.drawChessboardCorners(im_right_vis, (9,6), corners, ret) 
# plt.figure(figsize=(8,8))
# plt.imshow(im_right_vis)
# plt.show()


x,y=np.meshgrid(range(9),range(6)) 
world_points=np.hstack((x.reshape(54,1),y.reshape(54,1),np.zeros((54,1)))).astype(np.float32)
print(world_points)

_3d_points=[]
_2d_points=[]
img_paths=glob('./calibration_lg/*.png') #get paths of all all images     or   *.jpg
# de=0
for path in img_paths:
    im=cv2.imread(path)
    ret, corners = cv2.findChessboardCorners(im, (9,6))
#     de+=1
#     print(str(de)+") "+str(ret) +" --> " + path)
    if ret: 
        _2d_points.append(corners) #append current 2D points 
        _3d_points.append(world_points) #3D points are always the same
       
# corners=corners.reshape(-1,2) 
# im_right_vis = im.copy() # copy by value 
# cv2.drawChessboardCorners(im_right_vis, (9,6), corners, ret) 
# plt.figure(figsize=(8,8))
# plt.imshow(im_right_vis)
# plt.show()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(_3d_points, _2d_points, (im.shape[1],im.shape[0]),None, None)
print("Ret:",ret)
print("Mtx:",mtx," ----------------------------------> [",mtx.shape,"]") 
print("Dist:",dist," ----------> [",dist.shape,"]") 
print("rvecs:",rvecs," ------------------------------------------------- -------> [",rvecs[0].shape,"]")
print("tvecs:",tvecs," ------------------------------------------------- ------> [",tvecs[0].shape,"]")


##################### DISEGNO CUBO SU PATTERN #####################
# _3d_corners = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0], [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3]])
# image_index=4
# cube_corners_2d, _ = cv2.projectPoints(_3d_corners,rvecs[image_index],tvecs[image_index],mtx,dist)
#the underscore allows to discard the second output parameter (see doc)
# print(cube_corners_2d.shape) #the output consists in 8 2-dimensional poi nts

# img=cv2.imread(img_paths[image_index]) #load the correct image
# red=(0,0,255) #red (in BGR) 
# blue=(255,0,0) #blue (in BGR) 
# green=(0,255,0) #green (in BGR) 
# boo=(255,255,0)
# line_width=5
# #first draw the base in red
# cv2.line(img, tuple(cube_corners_2d[0][0]), tuple(cube_corners_2d[1][0 ]),red,line_width)
# cv2.line(img, tuple(cube_corners_2d[1][0]), tuple(cube_corners_2d[2][0 ]),red,line_width)
# cv2.line(img, tuple(cube_corners_2d[2][0]), tuple(cube_corners_2d[3][0 ]),red,line_width)
# cv2.line(img, tuple(cube_corners_2d[3][0]), tuple(cube_corners_2d[0][0 ]),red,line_width)
# #now draw the pillars
# cv2.line(img, tuple(cube_corners_2d[0][0]), tuple(cube_corners_2d[4][0 ]),blue,line_width)
# cv2.line(img, tuple(cube_corners_2d[1][0]), tuple(cube_corners_2d[5][0 ]),blue,line_width)
# cv2.line(img, tuple(cube_corners_2d[2][0]), tuple(cube_corners_2d[6][0 ]),blue,line_width)
# cv2.line(img, tuple(cube_corners_2d[3][0]), tuple(cube_corners_2d[7][0 ]),blue,line_width)
# #finally draw the top
# cv2.line(img, tuple(cube_corners_2d[4][0]), tuple(cube_corners_2d[5][0 ]),green,line_width)
# cv2.line(img, tuple(cube_corners_2d[5][0]), tuple(cube_corners_2d[6][0 ]),green,line_width)
# cv2.line(img, tuple(cube_corners_2d[6][0]), tuple(cube_corners_2d[7][0 ]),green,line_width)
# cv2.line(img, tuple(cube_corners_2d[7][0]), tuple(cube_corners_2d[4][0 ]),green,line_width)
# #cv2.line(img, tuple(start_point), tuple(end_point),(0,0,255),3) #we set the color to red (in BGR) and line width to 3
# plt.figure(figsize=(8,8)) 
# plt.imshow(img[...,::-1]) 
# plt.show()
##################### DISEGNO CUBO SU PATTERN #####################




im=cv2.imread("./calibration_lg/Screenshot 2020-04-12 at 19.40.22.png")[...,::-1] 
im_undistorted=cv2.undistort(im, mtx, dist) 
plt.figure(figsize=(8,8))
plt.subplot(121)
plt.imshow(im) 
plt.subplot(122) 
plt.imshow(im_undistorted) 
plt.show()
