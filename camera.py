"""!
Class to represent the camera.
"""

import cv2
import time
import numpy as np
import numpy.ma as ma
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

        #Frames indexed by (v,u) 
        self.VideoFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.GridFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720, 1280)).astype(np.uint16)
        self.DepthFrameProcessed = self.DepthFrameRaw.copy()
        self.DepthFrameZero = np.zeros_like(self.DepthFrameProcessed)
        self.redetect_mask = np.zeros_like(self.DepthFrameRaw, dtype=np.uint8)
        self.redetect_thresh = np.zeros_like(self.DepthFrameRaw, dtype=np.uint8)

        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.array([])
        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        self.intrinsic_matrix = np.array([[900.543212,0,655.99074785],[0,900.8950195,353.4480286],[0,0,1]])#np.array([])
        self.extrinsic_matrix = np.linalg.inv(np.array([[0.9994,-0.0349,0,0],[-0.0341,-.9776,-.2079,336.55],[0.0073,0.2078,-0.9781,990.6],[0,0,0,1]]))#np.array([])
        self.invIntrinsicCameraMatrix = np.linalg.inv(self.intrinsic_matrix)
        self.invExtrinsicCameraMatrix = np.linalg.inv(self.extrinsic_matrix)
        self.VectorUV = np.ones((3,1280*720))
        for iu in range(1280):
            for iv in range(720):
                self.VectorUV[0,iv+720*(iu-1)-1] = iu
                self.VectorUV[1,iv+720*(iu-1)-1] = iv
        self.VectorUVinCamera = self.invIntrinsicCameraMatrix.dot(self.VectorUV)
        self.last_click = np.array([0, 0])
        self.new_click = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.tag_detections = np.array([])
        self.tag_locations = [[-250.,-25.,0.],[250., -25.,0.], [250., 275.,0.],[-250.,275.,0.], [475.,-100., 155.], [-375.,400., 245.], [75.,200.,62.5], [-475.,-150.,95.]]
        self.dist_coeffs = np.array([0.125834,	-0.211044,	-0.001465,	0.00176,	0])#np.array([.140,-.459,-.001,0,0.405])#
        self.points_collected = 0

        self.bar_location = np.zeros((4,2))
        self.robot_sleep_loc = np.zeros((2,2))
        self.calibrated = False

        #Block Detection Info
        self.thresh = np.zeros_like(self.DepthFrameRaw)
        self.centroids = None
        self.contours = None
        self.homography = None
        self.TopThresh = 500
        self.BottomThresh = 10
        self.corner_coords_pixel = None
        self.gridUL = None 
        self.gridLR = None
        self.blockFrames = None
        self.detect = False
        self.blockData = [] #Should be list of lists, inner list have structure of [WorldFrame, color, theta, w, h, block_type] 

        #Debuggin nums
        self.TotalCt = 0.
        self.ZeroCt = 0.

        #BGR format
        self.colors_BGR = list(({'id': 'red', 'color': (10, 10, 127)},
                                {'id': 'orange', 'color': (30, 75, 150)},
                                {'id': 'yellow', 'color': (30, 150, 200)}, 
                                {'id': 'green', 'color': (60, 95, 35)},
                                {'id': 'blue', 'color': (100, 50, 0)},
                                {'id': 'violet', 'color': (100, 40, 80)},
                                {'id': 'pink', 'color': (203,192,255)}))

        self.color_font = {'red': (10, 10, 127),
                            'orange': (30, 75, 150),
                            'yellow': (30, 150, 200),
                            'green': (60, 95, 35),
                            'blue': (100, 50, 0),
                            'violet': (100, 40, 80),
                            'pink': (203,192,255)}

        self.color_contrast = {'red': 'green',
                                'green': 'red',
                                'violet': 'yellow',
                                'yellow':'violet',
                                'orange': 'blue',
                                'blue': 'orange',
                                'pink': 'yellow'}   
        
        #LAB format
        self.colors_LAB = list(({'id': 'red', 'color': (155, 145)},
                                {'id': 'orange', 'color': (160, 170)},
                                {'id': 'yellow', 'color': (130, 180)}, 
                                {'id': 'green', 'color': (105, 140)},
                                {'id': 'blue', 'color': (128, 100)},
                                {'id': 'violet', 'color': (135, 105)}))

        #Homography calibration
        self.corners_collected = 0
        self.corner_pts = np.zeros((4,2))
        #Setup for grid projection
        ypos = 50.0 * np.arange(-2.5, 9.5, 1.0)
        xpos = 50.0 * np.arange(-9.0, 10.0, 1.0)
        self.board_points = np.array(np.meshgrid(xpos, ypos)).T.reshape(-1, 2)  

        #Block Labeling {'block_type': (width, height)} width will be given the smaller value, height will be given the larger value
        self.block_types = list(({'id': 'Big Block', '(w,h)': (37.5, 37.5)},
                                {'id': 'Small Block', '(w,h)': (25, 25)},
                                {'id': 'Semi Circle', '(w,h)': (17, 35)},
                                {'id': 'Arch', '(w,h)': (28,57)}))

    def WorldtoPixel(self, world_coord):
        """!
        @brief      Convert world coordinates to pixel coordinates, returns (u,v)
        """        
        world_pos = np.ones((4,1))
        world_pos[0:3,0] = world_coord  
        camera_coords = np.matmul(self.extrinsic_matrix,world_pos)  
        Zc = camera_coords[2]
        return 1/Zc*np.matmul(self.intrinsic_matrix,camera_coords[0:3].reshape((3,1)))[0:2].reshape(1,2)

    def PixeltoWorldPos(self, u, v):

        pxFrame = np.array([[u],[v],[1]])

        camFrame = np.zeros((4,1))
        camFrame[0:3] = self.DepthFrameRaw[v,u]*np.matmul(self.invIntrinsicCameraMatrix,pxFrame)
        camFrame[-1] = 1

        worldPos = np.matmul(self.invExtrinsicCameraMatrix, camFrame)

        return worldPos

    def PxFrame2WorldFrame(self, pixel_pos, theta):
        """!
        @brief      Expresses Pixel Frame to World Frame
        """                

        #Construct uv1
        pxFrame = np.ones((3,1))
        u = pixel_pos[0]
        v = pixel_pos[1]
        pxFrame[0,0] = u
        pxFrame[1,0] = v
        Zc = self.DepthFrameRaw[v,u]
        
        camFrame = np.zeros((4,4))
        #Position in cameraFrame
        camFrame[0:3] = Zc*np.matmul(self.invIntrinsicCameraMatrix,pxFrame)
        #Place orientation of block in camera Frame, can optionally place this in worldframe instead since we dont account for skew anyways
        camFrame[0:3,0:3] = np.array([[np.cos(theta), -np.sin(theta), 0],
                                       [np.sin(theta), np.cos(theta), 0],
                                       [0, 0, 1]])
        camFrame[3,3] = 1

        worldFrame = np.matmul(self.invExtrinsicCameraMatrix, camFrame)

        return worldFrame

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
        #self.DepthFrameHSV[..., 0] = 6*self.redetect_thresh.astype(np.uint16) >> 1
        #self.DepthFrameHSV[..., 0] = 6*self.redetect_mask.astype(np.uint16) >> 1
        self.DepthFrameHSV[..., 0] = 6*self.thresh.astype(np.uint16) >> 1
        #self.DepthFrameHSV[..., 0] = 6*self.DepthFrameProcessed.astype(np.uint16) >> 1
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
                #frame = cv2.warpPerspective(frame, self.homography,(frame.shape[1], frame.shape[0]))
                #print('homographacation')      
                pass

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

    def retrieve_area_color(self, data, contour, colorspace):
        """!
        @brief      Utility function to help @c blockDetector() detect colors
        """         

        mask = np.zeros(data.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [contour], -1, 255, -1)
        if colorspace == "BGR":
            labels = self.colors_BGR
            mean = cv2.mean(data, mask=mask)[:3]
        elif colorspace == "LAB":
            labels = self.colors_LAB        
            mean = cv2.mean(data, mask=mask)[:3][1:]
        min_dist = (np.inf, None)
        for label in labels:
            d = np.linalg.norm(label["color"] - np.array(mean))
            if d < min_dist[0]:
                min_dist = (d, label["id"])
        return min_dist[1], mean

    def block_label(self, minRect):
        wh = minRect[1]
        if wh[0] > wh[1]:
            width, height = wh[1], wh[0]
        else:
            width, height = wh[0], wh[1]
        labels = self.block_types
        min_dist = (np.inf, None)        
        for label in labels:
            d = np.linalg.norm(label['(w,h)'] - np.array([width, height]))
            if d < min_dist[0]:
                min_dist = (d, label["id"])
        if min_dist[0] > 5:
            return 'Distractor'
        return min_dist[1]

    def block_height(self, data, contour):
        """!
        @brief      Utility function to determine depth of top of block
        """
        if data == "DepthFrameProcessed":
            depth = self.DepthFrameProcessed.copy()
        elif data == "DepthFrameRaw":
            depth = self.DepthFrameRaw.copy()
        mask = np.ones_like(self.DepthFrameRaw, dtype=np.uint8)
        v1, u1 = self.gridUL[1], self.gridUL[0]
        v2, u2 = self.gridLR[1], self.gridLR[0]
        mask[v1:(v2+1),u1:(u2+1)] -= 1
        readTheseVals = ma.masked_array(depth,mask = mask)

        x,y,w,h = cv2.boundingRect(contour)
        xmin = x
        xmax = x+w
        ymin = y
        ymax = y+h
        
        UR, LL = (xmin,ymin), (xmax,ymax)

        if data == "DepthFrameProcessed":
            height = np.amax(readTheseVals[ymin:ymax, xmin:xmax])
        elif data == "DepthFrameRaw":
            height = np.amin(readTheseVals[ymin:ymax, xmin:xmax])
        return UR, LL, height

    def block_redetect(self, data, contour):
        self.redetect_mask = np.zeros_like(self.DepthFrameRaw, dtype=np.uint8)

        UR, LL, height = self.block_height(data, contour)
    
        cv2.rectangle(self.redetect_mask, UR, LL, 255, cv2.FILLED)

        thresh = cv2.bitwise_and(cv2.inRange(self.DepthFrameProcessed, height-15, self.TopThresh), self.redetect_mask)

        kernel = np.ones((4,4), dtype = np.uint8)

        self.redetect_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        _, contour, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contour

    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """        
        font = cv2.FONT_HERSHEY_SIMPLEX
        contour_frame = self.VideoFrame
        VFcopy = self.VideoFrame.copy()
        #TODO: Draw labels on something other than DepthFrameRGB, VideoFrame refreshes too fast or something so it doesnt work that well there :/

        self.blockData = []
        for contour in self.contours:
            #Rerun block detection masking for just this contour to get data only on top block in stack
            contour = self.block_redetect("DepthFrameProcessed", contour)[0]
            VideoFrameLAB = cv2.cvtColor(VFcopy, cv2.COLOR_RGB2LAB)
            contour_color, LAB = self.retrieve_area_color(VideoFrameLAB, contour, "LAB")
    
            rect = cv2.minAreaRect(contour)
            wh = rect[1]
            theta = rect[2]
            block_type = self.block_label(rect)

            box = np.int0(cv2.boxPoints(rect))

            M = cv2.moments(contour)
            #self.TotalCt += 1.
            #self.percZero = self.ZeroCt/self.TotalCt*100
            #print("Percentage Zero: %.2f" %self.percZero)
            if cv2.contourArea(contour) != 0.0:
                font_color = self.color_font[self.color_contrast[contour_color]]
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                blockFrame = self.PxFrame2WorldFrame([cx,cy],theta)
                cv2.putText(contour_frame, contour_color, (cx-20, cy+40), font, 0.75,font_color , thickness=2)
                cv2.putText(contour_frame, str(int(theta)), (cx-20, cy), font, 0.5, font_color, thickness=2)
                cv2.putText(contour_frame, block_type, (cx-40, cy-50), font, 0.5, font_color, thickness=2)
                #cv2.putText(contour_frame, "LAB: %0.1f, %0.1f" %LAB, (cx-20, cy-50), font, 0.5, (0,0,0), thickness=2)
                cv2.drawContours(contour_frame,[box],0,font_color,2)
                try:
                    _, _, height = self.block_height("DepthFrameProcessed", contour)
                    cv2.putText(contour_frame, "Height: %.2f" %height, (cx-40, cy-30), font, 0.5, font_color, thickness=2)
                except:#Issues with empty contours i think
                    print("Issue retrieving height")                    
                data = [blockFrame, contour_color, theta, wh[0], wh[1], block_type]
                self.blockData.append(data)

            else:
                #self.ZeroCt +=1.
                pass
        #print(self.blockData)
        contour_frame = cv2.cvtColor(contour_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite("block_labels.png",contour_frame)                
                
    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """        

        lower = self.BottomThresh
        upper = self.TopThresh

        #Mask out robot arm
        mask = np.zeros_like(self.DepthFrameRaw, dtype=np.uint8)

        if self.calibrated:
            #Grid Mask
            cv2.rectangle(mask, self.gridUL, self.gridLR, 255, cv2.FILLED)
            #cv2.rectangle(self.VideoFrame,self.gridUL, self.gridLR, (255, 0, 0), 2)

            #Bar Masks
            cv2.rectangle(mask, tuple(self.bar_location[0,:]), tuple(self.bar_location[1,:]), 0, cv2.FILLED)
            cv2.rectangle(mask, tuple(self.bar_location[2,:]), tuple(self.bar_location[3,:]), 0, cv2.FILLED)
            #cv2.rectangle(self.VideoFrame,tuple(self.bar_location[0,:]), tuple(self.bar_location[1,:]), (255, 0, 0), 2)
            #cv2.rectangle(self.VideoFrame,tuple(self.bar_location[2,:]), tuple(self.bar_location[3,:]), (255, 0, 0), 2)

            #Robot Masks
            cv2.rectangle(mask, tuple(self.robot_sleep_loc[0,:]), tuple(self.robot_sleep_loc[1,:]), 0, cv2.FILLED)
            #cv2.rectangle(self.VideoFrame,tuple(self.robot_sleep_loc[0,:]), tuple(self.robot_sleep_loc[1,:]), (255, 0, 0), 2)

            self.thresh = cv2.bitwise_and(cv2.inRange(self.DepthFrameProcessed, lower, upper), mask)

            #Morphological Filter
            kernel = np.ones((8,8), dtype = np.uint8)
            #self.thresh = cv2.morphologyEx(self.thresh, cv2.MORPH_CLOSE, kernel)

            self.thresh = cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, kernel)

            _, self.contours, _ = cv2.findContours(self.thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
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

    def block_in_zone(self,zone):
        #Takes two world points that define a zone. If any contours are detected, then the zone is determined to have a block in it
        UR = self.WorldtoPixel(zone[0]).astype(int)[0]

        LL = self.WorldtoPixel(zone[1]).astype(int)[0]
 
        UR = tuple(UR)
        LL = tuple(LL)

        mask = np.zeros_like(self.DepthFrameRaw, dtype=np.uint8)
        cv2.rectangle(mask, UR, LL, 255, cv2.FILLED)
        thresh = cv2.bitwise_and(cv2.inRange(self.DepthFrameProcessed, 15, 100), mask)

        _, contour,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contour == []:
            return False
        else:
            print('Block in zone')
            return contour

    def contour_id(self, ref_contour):
        #Given a reference contour, find the block that is closest to that contour
        rect = cv2.minAreaRect(ref_contour)
        xy_ref = np.array(list(rect[0]))
        min_dist = (np.inf, None)        
        for idx, block in enumerate(self.blockData):
            blockFrame = block[0]
            xy = blockFrame[-1,0:2]
            d = np.linalg.norm(xy_ref - xy)
            if d < min_dist[0]:
                min_dist = (d, idx)
        return self.blockData[min_dist[1]]

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
        #self.camera.VideoFrame = cv2.undistort(cv_image,self.camera.intrinsic_matrix, self.camera.dist_coeffs, None)

        if self.camera.detect:
            self.camera.detectBlocksInDepthImage()
            self.camera.blockDetector()

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
        self.getFirst = True
    def callback(self, data):
        if self.getFirst:
            self.camera.intrinsic_matrix = np.reshape(data.K, (3, 3))
            #h, w = self.camera.VideoFrame.shape[:2]
            #self.camera.intrinsic_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera.intrinsic_matrix, self.camera.dist_coeffs, (w,h), 1, (w,h))
            self.getFirst = False
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
        self.camera.DepthFrameRaw += self.camera.DepthFrameZero.astype('uint16')
        #fself.camera.DepthFrameRaw = cv2.undistort(cv_depth,self.camera.intrinsic_matrix, self.camera.dist_coeffs, None)

        DepthFrameVector = self.camera.DepthFrameRaw.T.reshape((1,1280*720))
        CartesianInCamera = DepthFrameVector*self.camera.VectorUVinCamera
        WorldPointZs = self.camera.invExtrinsicCameraMatrix[2,:].dot(np.concatenate((CartesianInCamera,np.ones((1,1280*720))),axis=0))
        self.camera.DepthFrameProcessed = WorldPointZs.reshape((1280,720)).T
        #self.camera.DepthFrameProcessed -= self.camera.DepthFrameZero
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
