#! usr/bin/env/python3
import os
from cv2 import cv2
import numpy as np
import glob
import pandas as pd
import rospkg

rospack = rospkg.RosPack()
path = rospack.get_path('autonomous_turtlebot3')

# Bad pictures are the ones in which we cant detect all corners of chessboard
bad_pictures_list = open(path + "/calib/bad_pictures_list.txt", 'w')

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((5*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:5, 0:6].T.reshape(-1, 2)

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
images = glob.glob(path+'/calib/chessboard_images/*.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lwr = np.array([0, 0, 0])
    upr = np.array([255, 0, 0])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    msk = cv2.inRange(hsv, lwr, upr)
    msk = cv2.bitwise_not(msk)
    ret, corners = cv2.findChessboardCorners(msk, (5, 6), None)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
    # Store the non-feasible images
    if ret is False:
        bad_pictures_list.write(str(fname) + os.linesep)

img = cv2.imread(path + '/calib/chessboard_images/18a.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

h,  w = img.shape[:2]

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray_img.shape[::-1], None, None)

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
    mtx, dist, (w, h), 0, (w, h))

img = cv2.undistort(img, mtx, dist, None, newcameramtx)

cv2.imwrite('calib/undistorted.png', img)

# Store all matrices that we get from calibration on csv files
pd.DataFrame(data=newcameramtx.astype(float)).to_csv(
    path + "/calib/newcameramtx.csv", sep=' ', index=False, float_format='%.2f', header=False)
pd.DataFrame(data=mtx.astype(float)).to_csv(path + "/calib/mtx.csv", sep=' ',
                                            index=False, float_format='%.2f', header=False)
pd.DataFrame(data=dist.astype(float)).to_csv(path + "/calib/dist.csv", sep=' ',
                                             index=False, float_format='%.2f', header=False)
