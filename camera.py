"""!
Class to represent the camera.
"""

import cv2
import time
import numpy as np
from PyQt4.QtGui import QImage
from PyQt4.QtCore import QThread, pyqtSignal, QTimer
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from apriltag_ros.msg import *
from cv_bridge import CvBridge, CvBridgeError

#Stuff for finding countours
import argparse
import sys


class Camera():
    """!
    @brief      This class describes a camera.
    """
    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.GridFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720, 1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.array([])

        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        self.intrinsic_matrix = np.array([[900.543212,0,655.99074785],[0,900.8950195,353.4480286],[0,0,1]])#np.array([])
        self.extrinsic_matrix = np.linalg.inv(np.array([[0.9994,-0.0349,0,0],[-0.0341,-.9776,-.2079,336.55],[0.0073,0.2078,-0.9781,990.6],[0,0,0,1]]))#np.array([])
        self.last_click = np.array([0, 0])
        self.new_click = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.tag_detections = np.array([])
        self.tag_locations = [[-250,-25,0],[250, -25,0], [250, 275,0],[-250,275,0], [475,-100, 155], [-375,400, 245], [75,200,62.5], [-475,-150,95]]
        self.dist_coeffs = np.array([.140,-.459,-.001,0,0.405])
        """ block info """
        self.centroids = None
        self.block_detections = np.array([]) #unused
        self.homography = None
        self.TopThresh = 950
        self.BottomThresh = 987

        #Setup for grid projection
        ypos = 50.0 * np.arange(-2.5, 9.5, 1.0)
        xpos = 50.0 * np.arange(-9.0, 10.0, 1.0)
        xloc, yloc = np.meshgrid(xpos, ypos)
        self.board_points = np.array(np.meshgrid(xpos, ypos)).T.reshape(-1, 2)        

    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtGridFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """
        try:
            frame = cv2.resize(self.GridFrame, (1280, 720))

            if self.homography is not None:
                frame = cv2.warpPerspective(frame, self.homography,(frame.shape[1], frame.shape[0]))
                #print('homographacation')      

            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)      
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        pass

    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """        
        


    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """        

        lower = self.TopThresh
        upper = self.BottomThresh

        #Mask out robot arm
        mask = np.zeros_like(self.DepthFrameRaw, dtype=np.uint8)
        cv2.rectangle(mask, (275,120),(1100,720), 255, cv2.FILLED)
        cv2.rectangle(mask, (600,414),(740,720), 0, cv2.FILLED)
        
        cv2.rectangle(self.VideoFrame,(275,120),(1100,720), (255, 0, 0), 2)
        cv2.rectangle(self.VideoFrame, (600,414),(740,720), (255, 0, 0), 2)

        kernel = np.ones((6,6), dtype = np.uint8)

        #For stacked blocks, use np.max() to capture highest depth value of range, something like DepthFrameRaw[ymin:ymax, xmin:xmax]

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        thresh = cv2.bitwise_and(cv2.inRange(self.DepthFrameRaw, lower, upper), mask)



        cv2.imwrite("thresh.png",thresh) #Will write thresh to image that we can see, overwrites continuously but thats fine

        _, self.contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centr = []
        cv2.drawContours(self.DepthFrameRGB, self.contours, -1, (0,255,255), 3)        
        
        # for contour in self.contours:
        
        
        #     rect = cv2.minAreaRect(contour)
        #     box = cv2.boxPoints(rect)
        #     box = np.int0(box)<
        #     cv2.drawContours(self.DepthFrameRGB, [box], 0, (0,0,255), 2)

        #     print("Rectangle data")
        #     print(rect)
        #     print("")

            # Centroid Calculation using moment equations on contours   
            #M = cv2.moments(contour)        
            # if M["m00"] != 0.0:
            #     thisCenter = M["m10"] / M["m00"] ,M["m01"] / M["m00"]
            #     centr.append(thisCenter)  #TODO:Problematic because errors out when m00 is zero. May cause future bugs 
            #     cv2.circle(self.DepthFrameRGB, (int(thisCenter[0]), int(thisCenter[1])), 5, (255,0,255), 1)            
            
        #self.centroids = centr
        # print(self.centroids)
        # print('\n')


    def set_TopThresh(self,top_thresh):
        self.TopThresh = top_thresh
    def set_BottomThresh(self,bottom_thresh):
        self.BottomThresh = bottom_thresh

    def projectGridInRGBImage(self):
        """!
        @brief      projects

                    TODO: Use the intrinsic and extrinsic matricies to project the gridpoints 
                    on the board into pixel coordinates. copy self.VideoFrame to self.GridFrame and
                    and draw on self.GridFrame the grid intersection points from self.grid_points
                    (hint: use the cv2.circle function to draw circles on the image)
        """
        self.GridFrame = self.VideoFrame.copy()

        board_points_3D = np.column_stack((self.board_points, np.zeros(self.board_points.shape[0])))
        board_points_homogenous = np.column_stack((board_points_3D, np.ones(self.board_points.shape[0])))

        for point in board_points_homogenous:
            camera_coords = np.matmul(self.extrinsic_matrix,point).reshape(4)
            Zc = camera_coords[2]
            pixel_coords = 1/Zc*np.matmul(self.intrinsic_matrix,camera_coords[0:3])[0:2]
            #print(pixel_coords)
            self.GridFrame = cv2.circle(self.GridFrame, (int(pixel_coords[0]), int(pixel_coords[1])), 5, (0,0,255), 1)


class ImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image


class TagImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        self.camera.TagImageFrame = cv_image


class TagDetectionListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, AprilTagDetectionArray,
                                        self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.tag_detections = data
        #for detection in data.detections:
        # print(self.camera.tag_detections.detections[0])
        #print(detection.pose.pose.pose.position)


class CameraInfoListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, CameraInfo, self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.K, (3, 3))
        #print(self.camera.intrinsic_matrix)


class DepthListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
            
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth

        #self.camera.DepthFrameRaw = self.camera.DepthFrameRaw/2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage,QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_image_topic = "/tag_detections_image"
        tag_detection_topic = "/tag_detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        tag_image_listener = TagImageListener(tag_image_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Grid window", cv2.WINDOW_NORMAL)            
            time.sleep(0.5)
        while True:
            rgb_frame = self.camera.convertQtVideoFrame()
            depth_frame = self.camera.convertQtDepthFrame()
            tag_frame = self.camera.convertQtTagImageFrame()
            self.camera.projectGridInRGBImage()
            self.camera.detectBlocksInDepthImage()
            grid_frame = self.camera.convertQtGridFrame()
            if ((rgb_frame != None) & (depth_frame != None)):
                self.updateFrame.emit(rgb_frame, depth_frame, tag_frame, grid_frame)
            time.sleep(0.03)
            if __name__ == '__main__':
                cv2.imshow(
                    "Image window",
                    cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                cv2.imshow(
                    "Tag window",
                    cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                cv2.imshow("Grid window",
                    cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR))

                cv2.waitKey(3)
                time.sleep(0.03)


if __name__ == '__main__':
    camera = Camera()
    videoThread = VideoThread(camera)
    videoThread.start()
    rospy.init_node('realsense_viewer', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
