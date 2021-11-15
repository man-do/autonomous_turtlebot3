#! usr/bin/env python3

import cv2
import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


bridge = CvBridge()


def callback(image):
	cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
	rospy.loginfo(rospy.get_caller_id() + "I heard %s", cv_image.shape)
	cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
	cv_image = cv_image[140:240, :]
	# Select the point which define the plane which we want
	pts1 = ((0, 0), (320, 0), (0, 100), (320, 100)) 
	pts2 = ((0, 0), (520, 0), (200, 100), (320, 100))
	pts1 = np.float32(pts1)
	pts2 = np.float32(pts2)
	matrix = cv2.getPerspectiveTransform(pts1,pts2)
	dst = cv2.warpPerspective(cv_image,matrix,(520, 100))
	dim = (500, 600)
	# resize image
	resized = cv2.resize(dst, dim, interpolation = cv2.INTER_AREA)
	cv2.imshow('Image view', resized)
	cv2.waitKey(27)

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('/camera/image', Image, callback)
    rospy.spin()


if __name__ == '__main__':
    listener()
