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
idx = [1,3,4,5,6,7,8,9,10]
valid_idxs = []




x,y=np.meshgrid(range(9),range(6)) 
world_points=np.hstack((x.reshape(54,1),y.reshape(54,1),np.zeros((54,1 )))).astype(np.float32)
print(world_points)

_3d_points=[]
_2d_points=[]
img_paths=glob('./calibrazione_mac/*.jpg') #get paths of all all images 
# de=0
for path in img_paths:
    im=cv2.imread(path)
    ret, corners = cv2.findChessboardCorners(im, (9,6))
    # de+=1
    # print(str(de)+") "+str(ret) +" --> " + path)
    if ret: 
        _2d_points.append(corners) #append current 2D points 
        _3d_points.append(world_points) #3D points are always the same
       


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(_3d_points, _2d_points, (im.shape[1],im.shape[0]),None, None)
print("Ret:",ret)
print("Mtx:",mtx," ----------------------------------> [",mtx.shape,"]") 
print("Dist:",dist," ----------> [",dist.shape,"]") 
print("rvecs:",rvecs," ------------------------------------------------- -------> [",rvecs[0].shape,"]")
print("tvecs:",tvecs," ------------------------------------------------- ------> [",tvecs[0].shape,"]")




#####################  STEREO SYSTEM  #####################
for i in idx:
    im_left = cv2.imread("./stereo_img/left%02d.jpg"%i)
    im_right = cv2.imread("./stereo_img/right%02d.jpg"%i)

    ret_left,left_corners = cv2.findChessboardCorners(im_left, (9,6))
    ret_right,right_corners = cv2.findChessboardCorners(im_right, (9,6))

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
###############  FINE  ###################

# ##############  COSE AGGIUNTIVE  ##################
# selected_image = 2
# left_im=cv2.imread("./stereo_img/left03.jpg")
# right_im=cv2.imread("./stereo_img/right03.jpg")
# left_corners=all_left_corners[selected_image].reshape(-1,2)
# right_corners=all_right_corners[selected_image].reshape(-1,2)

# cv2.circle(left_im,(left_corners[0,0],left_corners[0,1]),10,(0,0,255),3)
# cv2.circle(right_im,(right_corners[0,0],right_corners[0,1]),10,(0,0,255),3)

# lines_right = cv2.computeCorrespondEpilines(all_left_corners[selected_image], 1,F)
# lines_right_F2 = cv2.computeCorrespondEpilines(all_left_corners[selected_image], 1,F2)

# print(lines_right.shape)
# lines_right = lines_right.reshape(-1,3)
# print(lines_right.shape)
# lines_right_F2 = lines_right_F2.reshape(-1,3)


# def drawLine(line,image,line_color):
#     a = line[0]
#     b = line[1]
#     c = line[2]

#     x0,y0 = map(int,[0, -c/b])
#     x1,y1 = map(int, [image.shape[1], -(c+a*image.shape[1])/b])

#     cv2.line(image,(x0,int(y0)),(x1,int(y1)),line_color,3)


# drawLine(lines_right[0],right_im,(0,255,255))

# R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx, dist, mtx, dist, (left_im.shape[1],left_im.shape[0]), R, T)
# print(right_im.shape[1])
# plt.figure(figsize=(16,8))
# plt.subplot(121)
# plt.imshow(left_im[...,::-1])
# plt.subplot(122)
# plt.imshow(right_im[...,::-1])
# plt.show()
##############  COSE AGGIUNTIVE  ##################



#####################  STEREO SYSTEM  #####################







##################### DISEGNO CUBO SU PATTERN #####################
# _3d_corners = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0], [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3]])
# image_index=7
# cube_corners_2d, _ = cv2.projectPoints(_3d_corners,rvecs[image_index],tvecs[image_index],mtx,dist)
# #the underscore allows to discard the second output parameter (see doc)
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