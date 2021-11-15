#! usr/bin/env/python3
import numpy as np
from cv2 import cv2
import pandas as pd
import rospy
from sensor_msgs.msg import Image
from sensor_msgs import point_cloud2
from std_msgs.msg import String, Header
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
import rospkg
from cv_bridge import CvBridge, CvBridgeError
import math


class lane_to_pointcloud():
    def __init__(self):
        # Read the matrices needed for undistortion
        self.rospack = rospkg.RosPack()
        self.path = self.rospack.get_path('autonomous_turtlebot3')
        self.mtx = pd.read_csv(
            self.path+'/calib/mtx.csv', sep=' ', header=None)
        self.newcameramtx = pd.read_csv(
            self.path+'/calib/newcameramtx.csv', sep=' ', header=None)
        self.dist = pd.read_csv(
            self.path+'/calib/dist.csv', sep=' ', header=None)
        self.mtx = np.asarray(self.mtx)
        self.newcameramtx = np.asarray(self.newcameramtx)
        self.dist = np.asarray(self.dist)
        self.bridge = CvBridge()
        self.board_on_ground = cv2.imread(
            self.path+'/calib/chessboard_images/board_on_ground1.png')
        self.h_mtx = self.ipm_homography(self.board_on_ground)

        # subscriber
        self.sub = rospy.Subscriber("camera/image", Image, self.video_callback)
        # publisher
        self.pub = rospy.Publisher(
            'IPM_pointcloud', PointCloud2, queue_size=2)
        self.img_pub = rospy.Publisher('video_ipm', Image, queue_size=2)

    def ipm_homography(self, image):
        # Find homography matrix for ipm
        if image is None:
            rospy.loginfo(
                'The IPM homography matrix cannot be calculated because the image of the board on ground is None')
            return 0

        image = cv2.undistort(
            image, self.mtx, self.dist, None, self.newcameramtx)

        # Detect black
        lwr = np.array([0, 0, 0])
        upr = np.array([255, 30, 30])
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        msk = cv2.inRange(hsv, lwr, upr)
        msk = cv2.bitwise_not(msk)
        ret, corners = cv2.findChessboardCorners(msk, (3, 3),
                                                 flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
        if ret == False:
            rospy.loginfo(
                "Chessboard corners not found when calculating IPM Homography matrix")
        if ret == True:
            pts1_index = [0, 2, 6, 8]
            pts1 = corners[pts1_index]
            pts2 = np.arange(8, dtype='float32').reshape(4, 1, 2)

            pts2[0][0][0] = pts1[0][0][0]+190
            pts2[0][0][1] = pts1[0][0][1]+300

            pts2[1][0][0] = pts1[1][0][0]+190
            pts2[1][0][1] = pts1[1][0][1]+300

            pts2[2][0][0] = pts1[0][0][0]+190
            pts2[2][0][1] = pts1[0][0][1] - \
                distance_2_points(pts1[0][0], pts1[2][0])+300

            pts2[3][0][0] = pts1[1][0][0]+190
            pts2[3][0][1] = pts1[1][0][1] - \
                distance_2_points(pts1[1][0], pts1[3][0]) + 300
            return cv2.getPerspectiveTransform(pts1, pts2)

        return 0

    def video_callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        points = self.get_pointcloud(image)
        self.pub.publish(points)

    def get_pointcloud(self, image):
        try:
            dst = cv2.undistort(
                image, self.mtx, self.dist, None, self.newcameramtx)
            warped = cv2.warpPerspective(
                dst, self.h_mtx, (700, 500), None)
            hsv = cv2.cvtColor(warped, cv2.COLOR_RGB2HSV)

            # yellow = left lane
            # white = right lane

            # Boundaries of yellow color
            yellow_lower = np.array([25, 240, 50])
            yellow_upper = np.array([100, 255, 255])

            # Boundaries of white color
            white_lower = np.array([0, 0, 200])
            white_upper = np.array([5, 5, 255])

            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            white_mask = cv2.inRange(hsv, white_lower, white_upper)
            white_mask_flipped = cv2.flip(white_mask, 1)

            yellow_left_edge = emphasize_left_corner(yellow_mask)
            white_right_edge = cv2.flip(
                emphasize_left_corner(white_mask_flipped), 1)

            result = cv2.add(yellow_left_edge, white_right_edge)
            result = result[350:, 200:500]
            img_msg = self.bridge.cv2_to_imgmsg(result)
            self.img_pub.publish(img_msg)

            features = cv2.goodFeaturesToTrack(result, 500, 0.01, 0.1)
            features = np.int0(features)
            #result = cv2.line(result, (350, 0), (350, 500), (0, 0, 255))
            points = []
            for feature in features:
                w = feature[0][0]
                h = feature[0][1]
                # divide the values ratio pixel/m constants vertical and horizontal
                x = (250 - w)/474.02597
                y = (200 - h)/177.147
                z = 0.2

                # RGB
                # +0.08 ???
                point = [y, x, z]
                points.append(point)

            fields = [PointField('x', 0, PointField.FLOAT32, 1),
                      PointField('y', 4, PointField.FLOAT32, 1),
                      PointField('z', 8, PointField.FLOAT32, 1),
                      ]
            header = Header()
            header.frame_id = "base_scan"
            pointcloud2 = point_cloud2.create_cloud(header, fields, points)
            return pointcloud2

        except CvBridgeError:
            print(CvBridgeError)

    def main(self):
        rospy.spin()


def distance_2_points(point1, point2):
    delta_w = point1[0]-point2[0]
    delta_h = point1[1]-point2[1]
    return math.sqrt(math.pow(delta_w, 2) + math.pow(delta_h, 2))


def emphasize_left_corner(src):
    img = src
    for i in range(4):
        img = cv2.Sobel(img, cv2.CV_32FC1, 1, 0, ksize=5)
        img[img < 0] = 0
    return img


if __name__ == '__main__':
    try:
        rospy.init_node('lane_to_pointcloud', anonymous=False)
        lane2pcl = lane_to_pointcloud()
        lane2pcl.main()
    except rospy.ROSInterruptException:
        pass
