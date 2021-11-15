import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((5*6, 3), np.float32)
objp[:,:2] = np.mgrid[0:5,0:6].T.reshape(-1,2)

objpoints = []
imgpoints = []

images = glob.glob('*.png')


for fname in images:
	img = cv.imread(fname)
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	ret, corners = cv.findChessboardCorners(gray, (5, 6), ( None))
	print(ret)
	
	if ret == False:
		cv.imshow('', img)
		cv.waitKey(1000)

	if ret == True:
		objpoints.append(objp)
	
		corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
		imgpoints.append(corners2)

		cv.drawChessboardCorners(img, (5, 6), corners2, ret)
		#cv.imshow('img', img)
		#cv.waitKey(1000)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print(ret)

img1 = cv.imread('2.png')
h,  w = img1.shape[:2]
newcameramtx, roi=cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
dst = cv.undistort(img1, mtx, dist, None, newcameramtx)

#cap = cv.VideoCapture(0)

#while(True):
#	ret, frame = cap.read()
#	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#	
#	dst = cv.undistort(gray, mtx, dist, None, newcameramtx)	
#
#	cv.imshow('frame', dst)
#	if cv.waitKey(1) & 0xFF == ord('q'):
#		break
#
#cap.release()
#cv.destroyAllWindows()	

cv.imshow('calibrated', dst)

cv.waitKey(100000)






