"""!
The state machine that implements the logic.
"""
from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
import time
import numpy as np
import rospy
import cv2
import math
import sys
import os
from kinematics import IK_geometric_two

class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.previous_state = "idle"
        self.next_state = "idle"
        self.waypoints = [
            [-np.pi/2,       -0.5,      -0.3,            0.0,       0.0],
            [0.75*-np.pi/2,   0.5,      0.3,      0.0,       np.pi/2],
            [0.5*-np.pi/2,   -0.5,     -0.3,     np.pi / 2,     0.0],
            [0.25*-np.pi/2,   0.5,     0.3,     0.0,       np.pi/2],
            [0.0,             0.0,      0.0,         0.0,     0.0],
            [0.25*np.pi/2,   -0.5,      -0.3,      0.0,       np.pi/2],
            [0.5*np.pi/2,     0.5,     0.3,     np.pi / 2,     0.0],
            [0.75*np.pi/2,   -0.5,     -0.3,     0.0,       np.pi/2],
            [np.pi/2,         0.5,     0.3,      0.0,     0.0],
            [0.0,             0.0,     0.0,      0.0,     0.0]]
        self.current_waypoint = 0
        self.waypoint_grip = [0,0,0,0,0,0,0,0,0,0] #Read 1 as closed and 0 as open
        self.err = 0
        self.thresh = 0.05
        self.long_time = 0
        self.record = 0
        self.pts_obtained = 0

        #Slow Calibration stuff

        self.image_points = np.ones((18,3))
        self.camera_points = np.zeros((18,3))
        self.world_points = 50*np.array([[-9., -2.5],
                                    [-9., 1.5],
                                    [-9., 4.5],
                                    [-9., 8.5],
                                    [-4., -2.5],
                                    [-4., 1.5],
                                    [-4., 4.5],
                                    [-4., 8.5],
                                    [0., 4.5],
                                    [0., 8.5],
                                    [4., -2.5],
                                    [4., 1.5],
                                    [4., 4.5],
                                    [4., 8.5],
                                    [9., -2.5],
                                    [9., 1.5],
                                    [9., 4.5],
                                    [9., 8.5]])        
        self.world_points = np.column_stack((self.world_points, np.zeros((self.world_points.shape[0],1))))

    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and functions as needed.
        """
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute(0)

        if self.next_state == "executeAndReturn":
            self.execute(1)

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()

        if self.next_state == "save_waypoint_close":
            self.save_waypoint_close()

        if self.next_state == "save_waypoint_open":
            self.save_waypoint_open()

        if self.next_state == "clear_waypoints":
            self.clear_waypoints()

        if self.next_state == "save_waypoints":
            self.save_waypoints()

        if self.next_state == "save_image":
            self.save_image()

        if self.next_state == "zero_depth":
            self.zero_depth()

        if self.next_state == "calibrate_slow":
            self.calibrate_slow()

        if self.next_state == "click_place":
            self.click_place()

        if self.next_state == "event1":
            self.event1()
    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self, returnFlag):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        if returnFlag:
            if self.current_state != "executeAndReturn":
                self.previous_state = self.current_state
                print("Previous State Equals")
                print(self.current_state)
                print('\n')
            self.current_state = "executeAndReturn"    
        else:
            print("THIS SHOULD NOT RUN")
            self.current_state="execute"
            self.previous_state = "idle"

        if self.current_waypoint == 0:
            self.now = time.time()
            self.start_exe = time.time()

        #Calculate Joint Errors
        joint_errors = np.array(self.waypoints[self.current_waypoint]) - np.array(self.rxarm.get_positions())
        self.err = np.sqrt(np.mean(joint_errors**2))

        #Waypoint checking and actuation
        if (self.err < self.thresh) or (self.current_waypoint == 0) or (self.long_time):
            self.rxarm.set_positions(self.waypoints[self.current_waypoint])          
            self.actuate_gripper(self.waypoint_grip[self.current_waypoint])
            self.current_waypoint += 1
            self.zTime = time.time()
            self.long_time = 0
        else:
            if (time.time() - self.zTime >3):
                self.long_time = 1

        #Record data
        if self.record:
            if (time.time()-self.now >=0.2):
                self.now = time.time()
                file1 = open("Data.txt","a")
                file1.write(str(self.now - self.start_exe) + ", " + str(self.waypoints[self.current_waypoint]) )
                file1.write('\n')
                file1.close()


        #Check if all waypoints have been passed through
        if self.current_waypoint == len(self.waypoints):
            self.current_waypoint = 0
            self.next_state = self.previous_state
            print("Next State Equals")
            print(self.next_state)
            print('\n')
        self.status_message = "State: Execute - Executing motion plan, Err =" + str(self.err) + "Time since cmd:" + str(time.time()-self.zTime)

    def calibrate(self):
        """!
        @brief      Auto calibration using april tags
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        image_points = np.zeros((8,3))
        model_points = np.zeros((8,3))

        """TODO Perform camera calibration routine here"""
        #--------------Intrinsic Matrix Correction--------------
        # h, w = self.camera.VideoFrame.shape[:2]
        # self.camera.intrinsic_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera.intrinsic_matrix, self.camera.dist_coeffs, (w,h), 1, (w,h))

        #--------------Extrinsic Matrix Calculation--------------
        #Read in April Tag data
        for idx, tag in enumerate(self.camera.tag_detections.detections):
            x = tag.pose.pose.pose.position.x
            y = tag.pose.pose.pose.position.y
            z = tag.pose.pose.pose.position.z
            image_points[idx,:] = (np.matmul(self.camera.intrinsic_matrix,np.array([x,y,z]).reshape(3,1))/z).reshape(3)
            model_points[idx,:] = np.array(self.camera.tag_locations[tag.id[0]-1])
        #Take only desired slices, and assert that data type is float32
        image_points=image_points[:,0:2].astype('float32')
        model_points=model_points.astype('float32')

        #Strips preallocated rows
        image_points = image_points[~np.all(image_points == 0, axis=1)]
        model_points = model_points[~np.all(model_points == 0, axis=1)]

        #Get translation and rotation vector
        (success,rot_vec,trans_vec) = cv2.solvePnP(model_points,
                                                   image_points,
                                                   self.camera.intrinsic_matrix,
                                                   None,
                                                   flags=cv2.SOLVEPNP_ITERATIVE)

        #Use Rodrigues to convert rotataion vector into rotation matrix
        theta = np.linalg.norm(rot_vec)
        if theta < sys.float_info.epsilon:              
            rotation_mat = np.eye(3, dtype=float)
        else:
            r = rot_vec/ theta
            I = np.eye(3, dtype=float)
            r_rT = np.array([
                [r[0]*r[0], r[0]*r[1], r[0]*r[2]],
                [r[1]*r[0], r[1]*r[1], r[1]*r[2]],
                [r[2]*r[0], r[2]*r[1], r[2]*r[2]]
                ])
            r_cross = np.array([
                [0, -r[2], r[1]],
                [r[2], 0, -r[0]],
                [-r[1], r[0], 0]
                ])
            r_rT = r_rT.reshape(3,3)
        rotation_mat = math.cos(theta) * I + (1 - math.cos(theta)) * r_rT + math.sin(theta) * r_cross
        #Put together extrinsic matrix
        spatial_transform = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]], dtype=np.float64)

        spatial_transform[0:3,3] = trans_vec.reshape(3)
        spatial_transform[0:3,0:3] = rotation_mat
        self.camera.extrinsic_matrix = spatial_transform

        self.camera.invIntrinsicCameraMatrix = np.linalg.inv(self.camera.intrinsic_matrix)
        self.camera.invExtrinsicCameraMatrix = np.linalg.inv(self.camera.extrinsic_matrix)
        self.camera.VectorUVinCamera = self.camera.invIntrinsicCameraMatrix.dot(self.camera.VectorUV)

        #--------------Homography Transform calculations--------------
        corner_coords_world = np.array([[500,-175,0],[500,475,0], [-500, 475,0], [-500,-175,0]]) #[LR, UR, UL, LL]
        corner_coords_pixel = np.zeros((4,2))
        for i in range(4):
            corner_coords_pixel[i,:] = self.camera.WorldtoPixel(corner_coords_world[i,:])

        """TODO Make calibration force user to click on corner points"""
        src_pts = corner_coords_pixel
        dest_pts = np.array([[1280,720], [1280,0], [0,0], [0,720]])

        self.camera.homography = cv2.findHomography(src_pts, dest_pts)[0]

        #Saving positions for block detection masking
        self.camera.gridUL = tuple(corner_coords_pixel[2,:].astype(int))
        self.camera.gridLR = tuple(corner_coords_pixel[0,:].astype(int))

        self.camera.gridUL_flip = tuple([self.camera.gridUL[1], self.camera.gridUL[0]])
        self.camera.gridLR_flip = tuple([self.camera.gridLR[1], self.camera.gridLR[0]])

        barloc_world = 50*np.array([[-11,7,0],[-9.5, 0,0], [11, 7,0],[9.5, 0,0]])
        robotsleep_world= 50*np.array([[-2.,3.,0.],[2., -3.5,0.]])
        for i in range(4):
            self.camera.bar_location[i,:] = self.camera.WorldtoPixel(barloc_world[i,:])
        for i in range(2):
            self.camera.robot_sleep_loc[i,:] = self.camera.WorldtoPixel(robotsleep_world[i,:])
        self.camera.bar_location = self.camera.bar_location.astype(int)
        self.camera.robot_sleep_loc = self.camera.robot_sleep_loc.astype(int) 

        self.camera.calibrated = True
        self.status_message = "Calibration - Completed Calibration"

    """ TODO """
    def detect(self):
        """!
        @brief      Detect the blocks
        """
        self.current_state = "detect"
        self.status_message = "Looking for blocks"
        # self.camera.detectBlocksInDepthImage()
        # if self.camera.contours is not None:
        #     self.camera.blockDetector()
        
        if self.camera.detect == True:
            self.camera.detect = False
        else:
            self.camera.detect = True
        
        self.next_state = "idle"

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            rospy.sleep(5)
        self.next_state = "idle"

    def save_waypoint_close(self):
        """!
        @brief      Adds waypoint with a closed gripper to set of waypoints
        """
        self.current_state = "save_waypoint_close"
        self.waypoint_grip.append(1)
        self.waypoints.append(self.rxarm.get_positions())
        self.next_state = "idle"

    def save_waypoint_open(self):
        """!
        @brief      Adds waypoint with an open gripper to set of waypoints
        """

        self.current_state = "save_waypoint_open"
        self.waypoint_grip.append(0)
        self.waypoints.append(self.rxarm.get_positions().tolist())
        self.next_state = "idle"

    def clear_waypoints(self):
        """!
        @brief      Clear set of waypoints currently in memory
        """
        self.current_state = "clear_waypoints"
        self.waypoints = []
        self.waypoint_grip = []
        self.current_waypoint = -1
        self.err = 0
        self.next_state = "idle"

    def save_waypoints(self):
        """!
        @brief      Export set of waypoints currently in memory to a .txt file
        """        
        self.current_state = "save_waypoints"
        self.status_message = "Saving waypoints"
        file1 = open("WP.txt","a")
        file1.write(str(self.waypoints))
        file1.write('\n')
        file1.write(str(self.waypoint_grip))
        file1.close()
        self.next_state = "idle"
        
    def save_image(self):
        """!
        @brief      Save camera image for asynchrnous CV testing
        """        
        
        self.current_state = "save_image"
        
        #Thresholded Depth Image
        cv2.imwrite("testing_thresh.png",self.camera.thresh)
        
        #Depth image as seen in gui
        DepthFrameBGR = cv2.cvtColor(self.camera.DepthFrameRGB, cv2.COLOR_RGB2BGR)
        cv2.imwrite("testing_DepthScreen.png", DepthFrameBGR)

        #Raw Depth Frame
        cv2.imwrite("testing_DepthRaw.png", self.camera.DepthFrameRaw)

        #RGB Image
        VidFrameBGR = cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR)
        cv2.imwrite("testing_RGB.png", VidFrameBGR)
        
        #Depth Frame with world coord
        cv2.imwrite("testing_DepthProc.png", self.camera.DepthFrameProcessed)

        self.next_state = "idle"
        
    def zero_depth(self):
        self.current_state = "zero_depth"
        self.camera.DepthFrameZero = self.camera.DepthFrameProcessed
        self.next_state = "idle"

    def calibrate_slow(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate_slow"
        """TODO Perform camera calibration routine here"""
        '''Extrinsics calculation
        user should click on some specified grid coordinates in a specific order ->this goes in image_points
        image points need to be backtracked into camera points using intrinsic matrix

        refer to stanford paper for how to solve for rotation and translation matrix given the camera points and world points
        '''
        if self.camera.points_collected <=18:
            if self.camera.new_click:
                self.image_points[self.camera.points_collected-1,0:2] = self.camera.last_click
                self.camera.new_click = False
            else:
                self.status_message = "Please click on: " + np.array2string(self.world_points[self.camera.points_collected], formatter={'float_kind':lambda x: "%.1f" % x})
            #Force it to move on once we have all points
            if self.camera.points_collected == 18:
                self.camera.points_collected +=1
        else:
            self.camera.points_collected = 0
            self.camera.invIntrinsicCameraMatrix = np.linalg.inv(self.camera.intrinsic_matrix)
            for i in range(self.image_points.shape[0]):
                uv1 = self.image_points[i,:]
                v = uv1[1]
                u = uv1[0]
                uv1 = uv1.reshape((3,1))
                Zc = self.camera.DepthFrameRaw[int(v),int(u)]
                self.camera_points[i,:] = (Zc*np.matmul(self.camera.invIntrinsicCameraMatrix,uv1)).reshape(3)

            camera_mean = np.mean(self.camera_points, axis = 0)
            world_mean = np.mean(self.world_points, axis = 0)

            correlation = np.zeros((world_mean.shape[0],camera_mean.shape[0]))
            for i in range(self.world_points.shape[0]):
                world_ci = self.world_points[i,:].reshape((3,1)) - world_mean.reshape((3,1))
                camera_ci = self.camera_points[i,:].reshape((3,1)) - camera_mean.reshape((3,1))
                correlation += np.matmul(world_ci, camera_ci.T)
            U_rot, s_rot, VT_rot = np.linalg.svd(correlation, full_matrices = False)
            R = np.matmul(VT_rot.T, U_rot.T)

            if abs(np.linalg.det(R)-1)<0.005:
                pass
            else:
                V = VT_rot.T
                V[:,2] *= -1
                R = np.matmul(V, U_rot.T)
                detR = np.linalg.det(R)
                print('Had to correct for planar points')
                # print(f'New determinant is {detR}')

            T = camera_mean.T - np.matmul(R, world_mean.T)
            
            self.camera.extrinsic_matrix = np.zeros((4,4))
            self.camera.extrinsic_matrix[-1,-1] = 1
            self.camera.extrinsic_matrix[:3,:3] = R
            self.camera.extrinsic_matrix[:3,-1] = T

            self.camera.invExtrinsicCameraMatrix = np.linalg.inv(self.camera.extrinsic_matrix)

            #--------------Homography Transform calculations--------------
            corner_coords_world = np.array([[500,-175,0],[500,475,0], [-500, 475,0], [-500,-175,0]]) #[LR, UR, UL, LL]
            corner_coords_pixel = np.zeros((4,2))
            for i in range(4):
                corner_coords_pixel[i,:] = self.camera.WorldtoPixel(corner_coords_world[i,:])

            """TODO Make calibration force user to click on corner points"""
            src_pts = corner_coords_pixel
            dest_pts = np.array([[1280,720], [1280,0], [0,0], [0,720]])

            self.camera.homography = cv2.findHomography(src_pts, dest_pts)[0]

            #Saving positions for block detection masking
            self.camera.gridUL = tuple(corner_coords_pixel[2,:].astype(int))
            self.camera.gridLR = tuple(corner_coords_pixel[0,:].astype(int))
            barloc_world = 50*np.array([[-11,7,0],[-9.5, 0,0], [11, 7,0],[9.5, 0,0]])
            robotsleep_world= 50*np.array([[-2.,3.,0.],[2., -3.5,0.]])
            for i in range(4):
                self.camera.bar_location[i,:] = self.camera.WorldtoPixel(barloc_world[i,:])
            for i in range(2):
                self.camera.robot_sleep_loc[i,:] = self.camera.WorldtoPixel(robotsleep_world[i,:])
            self.camera.bar_location = self.camera.bar_location.astype(int)
            self.camera.robot_sleep_loc = self.camera.robot_sleep_loc.astype(int) 

            self.camera.calibrated = True
            self.next_state = "idle"


            self.status_message = "Calibration - Completed Calibration"

    def click_place(self):
        self.current_state = "click_place"
        count = self.waypoint_grip[-1] 
        '''
        if points_coll < desired:
            if nnew click:
                points[click] = new click
                new click = False
        else:
            build waypoints and execute

        '''
        if self.camera.new_click:
            
            if count == 0:
                pt = self.camera.last_click
                ptW = self.camera.PixeltoWorldPos(pt[0],pt[1])
                ptW_approch = self.camera.PixeltoWorldPos(pt[0],pt[1])
                ptW[2] = ptW[2] -20
                ptW_approch[2] = ptW_approch[2] + 50
                theta_des = IK_geometric_two(self.rxarm.dh_params,ptW[0:3].reshape(3),'down')
                theta_des_approach = IK_geometric_two(self.rxarm.dh_params,ptW_approch[0:3].reshape(3),'down')

                print(count)
                
                self.waypoints = []
                self.waypoint_grip = []
                self.waypoints.append(theta_des_approach)
                self.waypoint_grip.append(0)

                self.waypoints.append(theta_des)
                self.waypoint_grip.append(0)

                self.waypoints.append(theta_des)
                self.waypoint_grip.append(1)

                self.waypoints.append(theta_des_approach)
                self.waypoint_grip.append(1)

                self.next_state = "execute"
                self.camera.new_click = False
        
            if count == 1:
                pt = self.camera.last_click
                ptW = self.camera.PixeltoWorldPos(pt[0],pt[1])
                ptW_approch = self.camera.PixeltoWorldPos(pt[0],pt[1])
                ptW[2] = ptW[2] + 50
                ptW_approch[2] = ptW_approch[2] + 100
                theta_des = IK_geometric_two(self.rxarm.dh_params,ptW[0:3].reshape(3),'down')
                theta_des_approach = IK_geometric_two(self.rxarm.dh_params,ptW_approch[0:3].reshape(3),'down')

                print(count)
                self.waypoints = []
                self.waypoint_grip = []

                self.waypoints.append(theta_des_approach)
                self.waypoint_grip.append(1)

                self.waypoints.append(theta_des)
                self.waypoint_grip.append(1)

                self.waypoints.append(theta_des)
                self.waypoint_grip.append(0)

                self.waypoints.append(theta_des_approach)
                self.waypoint_grip.append(0)

                self.next_state = "execute"
                self.camera.new_click = False

    def event1(self):
        print("RUNNING EVENT 1")
        self.current_state = "event1"
        count = self.waypoint_grip[-1] 
        '''
        if points_coll < desired:
            if nnew click:
                points[click] = new click
                new click = False
        else:
            build waypoints and execute

        '''
        if self.camera.new_click:
            
            if count == 0:
                pt = self.camera.last_click
                ptW = self.camera.PixeltoWorldPos(pt[0],pt[1])
                ptW_approch = self.camera.PixeltoWorldPos(pt[0],pt[1])
                ptW[2] = ptW[2] -20
                ptW_approch[2] = ptW_approch[2] + 50
                theta_des = IK_geometric_two(self.rxarm.dh_params,ptW[0:3].reshape(3),'down')
                theta_des_approach = IK_geometric_two(self.rxarm.dh_params,ptW_approch[0:3].reshape(3),'down')

                print(count)
                
                self.waypoints = []
                self.waypoint_grip = []
                self.waypoints.append(theta_des_approach)
                self.waypoint_grip.append(0)

                self.waypoints.append(theta_des)
                self.waypoint_grip.append(0)

                self.waypoints.append(theta_des)
                self.waypoint_grip.append(1)

                self.waypoints.append(theta_des_approach)
                self.waypoint_grip.append(1)

                self.next_state = "executeAndReturn"
                self.camera.new_click = False
        
            if count == 1:
                pt = self.camera.last_click
                ptW = self.camera.PixeltoWorldPos(pt[0],pt[1])
                ptW_approch = self.camera.PixeltoWorldPos(pt[0],pt[1])
                ptW[2] = ptW[2] + 50
                ptW_approch[2] = ptW_approch[2] + 100
                theta_des = IK_geometric_two(self.rxarm.dh_params,ptW[0:3].reshape(3),'down')
                theta_des_approach = IK_geometric_two(self.rxarm.dh_params,ptW_approch[0:3].reshape(3),'down')

                print(count)
                self.waypoints = []
                self.waypoint_grip = []

                self.waypoints.append(theta_des_approach)
                self.waypoint_grip.append(1)

                self.waypoints.append(theta_des)
                self.waypoint_grip.append(1)

                self.waypoints.append(theta_des)
                self.waypoint_grip.append(0)

                self.waypoints.append(theta_des_approach)
                self.waypoint_grip.append(0)

                self.next_state = "execute"
                self.camera.new_click = False
#--------------------------Helper Functions-------------------------------------------------#
    def actuate_gripper(self,grip_state):
        #Actuate Grippers
        if grip_state:
            self.rxarm.close_gripper()
        else:
            self.rxarm.open_gripper()
    


class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            rospy.sleep(0.05)