#! usr/bin/env/python3

from cv2 import cv2
from cv_bridge.core import CvBridge
import rospy
from autonomous_turtlebot3.lane_detector import LaneDetector
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField


class lane_detector_node():
    def __init__(self):
        self.bridge = CvBridge()
        self.detector = LaneDetector()
        # publisher
        self.pub = rospy.Publisher(
            'IPM_pointcloud', PointCloud2, queue_size=2)
        self.img_pub = rospy.Publisher('video_ipm', Image, queue_size=2)
        # subscriber
        self.sub = rospy.Subscriber("camera/image", Image, self.video_callback)

    def video_callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        points = self.detector.get_pointcloud(image)
        self.pub.publish(points)

    def main(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        rospy.init_node('lane_detector_node', anonymous=False)
        ldetector = lane_detector_node()
        ldetector.main()
    except rospy.ROSInterruptException:
        pass
