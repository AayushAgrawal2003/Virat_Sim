#!/usr/bin/python3
from __future__ import print_function
from sensor_msgs.msg import CameraInfo, Image
import rospy
import cv2
from cv_bridge import CvBridge
import numpy as np
import argparse
import random as rng
rng.seed(12345)
import struct
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
K = CameraInfo()

pub = rospy.Publisher("point_cloud2", PointCloud2, queue_size=2)

fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          # PointField('rgb', 12, PointField.UINT32, 1),
          PointField('rgba', 12, PointField.UINT32, 1),
          ]

header = Header()
header.frame_id = "map"


def start_node():
    rospy.init_node('detect_pump')
    rospy.Subscriber("rrbot/camera1/image_raw", Image, process_image)
    rospy.spin()


def showImage(img):
    cv2.imshow('image', img)
    cv2.waitKey(1)



def thresh_callback(val):
    threshold = val
    global src_gray
    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)

    one,contours,two = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    minEllipse = [None]*len(contours)
    for i, c in enumerate(contours):
        if c.shape[0] > 850:
            minEllipse[i] = cv2.fitEllipse(c)
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i, c in enumerate(contours):
        color = (255,255,255)
        
        if c.shape[0] > 38:
            cv2.drawContours(drawing, contours, i, color)
            cv2.ellipse(drawing, minEllipse[i], color, 2)

    #cv.imshow('Contours', drawing)
    return canny_output



def process_image(msg):
    try:
        # convert sensor_msgs/Image to OpenCV Image
        bridge = CvBridge()
        orig = bridge.imgmsg_to_cv2(msg, "bgr8")
        src = orig
        global src_gray
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        src_gray = cv2.blur(src_gray, (10,10))
        #rospy.loginfo(src_gray)

        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        lower_red = np.array([30,150,50])
        upper_red = np.array([255,255,180])
    
        mask = cv2.inRange(hsv, lower_red, upper_red)
        
        res = cv2.bitwise_and(src,src, mask= mask)
        max_thresh = 255
        thresh = 230
        #cv2.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
        #sec = thresh_callback(thresh)
        #blob detection
        

        params = cv2.SimpleBlobDetector_Params()
        

        params.filterByArea = True
        params.minArea = 10

        params.filterByInertia = True
        params.minInertiaRatio = 0.00001



        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(mask)
        blank = np.ones((4,4))
        blobs = cv2.drawKeypoints(src, keypoints, 100, (0, 0, 255),
                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )
        

            
        #cv2.imshow("Filtering Circular Blobs Only", src_gray)
        
        

        pts = ([key_point.pt for key_point in keypoints])
        
        #sshowImage(blobs)
        point_holder = []
        rgb = struct.unpack('I', struct.pack('BBBB',0, 255, 255, 100))[0]
        def call(msg):
            global K
            K = np.array(msg.K).reshape([3, 3])
            K = np.linalg.inv(K)

        
         # rotation matrix
        roll = 0
        pitch = -0.45
        yaw = 0
        h = 1.18
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        rotation_ground_to_cam = np.array([[cr*cy+sp*sr+sy, cr*sp*sy-cy*sr, -cp*sy],
                                         [cp*sr, cp*cr, sp],
                                         [cr*sy-cy*sp*sr, -cr*cy*sp -sr*sy, cp*cy]])

        rotation_cam_to_ground = rotation_ground_to_cam.T # inv of rotation mat is same as its transpose

        n = np.array([0, 1, 0])
        ground_normal_to_cam = (rotation_cam_to_ground.T).dot(n)  #nc
    
        main = rospy.Subscriber("rrbot/camera1/camera_info",CameraInfo,call)

        rospy.loginfo(K)
        
        for i in pts:
            vals = np.array([float(i[0]),float(i[1]),1]) 
            vector = h * np.dot(K,vals) * h / (np.dot(ground_normal_to_cam.T ,np.dot(K,vals)))
            rospy.loginfo(vector)
            point_holder += [[vector[0], vector[1],0.0, rgb]]
        
        
            

        #rospy.loginfo(point_holder)
        pc2 = point_cloud2.create_cloud(header, fields, point_holder)
        pc2.header.stamp = rospy.Time.now()
        pc2.header.frame_id = "odom"
        pub.publish(pc2)
        showImage(blobs)

        
    except Exception as err:
        print (err)



if __name__ == '__main__':
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass


