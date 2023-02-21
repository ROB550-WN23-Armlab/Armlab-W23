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
import matplotlib.pyplot as plt
from kinematics import IK_geometric_two, IK_geometric_event_1, get_R_from_euler_angles, FK_dh, get_pose_from_T

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
        self.thresh = 0.025
        self.long_time = 0
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

        self.sorted = False

        self.ROYGBV = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']

        self.ROYGBV_idx = {'red':0, 'orange':1, 'yellow':2, 'green':3, 'blue':4, 'violet':5}

        self.next_big_color = 'red'

        self.next_small_color = 'red'

        self.init1 = True
        self.init2 = True
        self.init3 = True
        self.init4 = True
        self.init_b = True
        self.smallStackCt2 = 0
        self.bigStackCt2 = 0

        self.event3small = {'red': -6.5, 
                            'orange': -5.5,
                            'yellow': -4.5, 
                            'green': -3.5, 
                            'blue': -2.5, 
                            'violet': -1.5}

        self.event3big =   {'red': 1.5, 
                            'orange': 2.5,
                            'yellow': 3.5, 
                            'green': 4.5, 
                            'blue': 5.5, 
                            'violet': 6.5}            

        self.event4small =  {'red': -5, 
                            'orange': -4.5,
                            'yellow': -4, 
                            'green': -3.5, 
                            'blue': -3, 
                            'violet': -2.5}

        self.event4bigX =    {'red': 2.5, 
                            'orange': 3,
                            'yellow': 3.5, 
                            'green': 4, 
                            'blue': 4.5, 
                            'violet': 5}         

        self.event4smallX =  {'red': -5, 
                            'orange': -4.5,
                            'yellow': -4, 
                            'green': -3.5, 
                            'blue': -3, 
                            'violet': -2.5}

        self.event4smallY = {'red': -2.5, 
                            'orange': -0.5,
                            'yellow': -2.5, 
                            'green': -0.5, 
                            'blue': -2.5, 
                            'violet': -0.5}

        self.event4bigY = {'red': -2.5, 
                            'orange': -0.5,
                            'yellow': -2.5, 
                            'green': -0.5, 
                            'blue': -2.5, 
                            'violet': -0.5}

        self.Three2Four = False    

        self.isGrabbingInFront = True      

        self.sizeTower = 1.0

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
            self.robot_home()            
            self.detect()

        if self.next_state == "manual":
            self.manual()

        if self.next_state == "save_waypoint_close":
            self.save_waypoint_close()

        if self.next_state == "save_waypoint_open":
            self.save_waypoint_open()

        if self.next_state == "clear_waypoints":
            self.clear_waypoints()

        if self.next_state == "save_trajectory":
            self.save_trajectory()

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

        if self.next_state == "event2":
            self.event2()

        if self.next_state == "event3":
            self.event3()

        if self.next_state == "event4":
            self.event4()            
    
        if self.next_state == "bonus":
            self.bonusRob()
    
        if self.next_state == "test":
            self.test()

        if self.next_state == "plot":
            self.plot()
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
            self.current_state = "executeAndReturn"    
        else:
            print("THIS SHOULD NOT RUN")
            self.current_state="execute"
            self.previous_state = "idle"

        if self.current_waypoint == 0:
            self.now = time.time()
            self.start_exe = time.time()
            self.zTime = time.time()

        #Calculate Joint Errors
        joint_errors = np.array(self.waypoints[self.current_waypoint]) - np.array(self.rxarm.get_positions())
        self.err = np.sqrt(np.mean(joint_errors**2))

        self.rxarm.set_moving_time(1.5)
        #Waypoint checking and actuation
        if (self.err < self.thresh) or (self.current_waypoint == 0) or (self.long_time):
            self.rxarm.set_positions(self.waypoints[self.current_waypoint])
            self.rxarm.set_gripper_pressure(self.waypoint_grip[self.current_waypoint])
            self.actuate_gripper((self.waypoint_grip[self.current_waypoint]>0))
            self.zTime = time.time()
            self.current_waypoint += 1
            self.long_time = 0
        else:
            if (time.time() - self.zTime >self.rxarm.moving_time):
                self.long_time = 1
                #Record data    
        if (time.time() - self.zTime >0.05):                
                self.now = time.time()
                pos = self.rxarm.get_positions().tolist()
                pos.append((self.waypoint_grip[self.current_waypoint])) #pos = [q1,q2,q3,q4,q5,grip_state]
                file1 = open("JointData.txt","a")
                file1.write(str(self.now - self.start_exe) + ", " + str(pos))
                file1.write('\n')
                file1.close()

                file2 = open("Data.txt","a")
                pos = get_pose_from_T(FK_dh(self.rxarm.dh_params, pos, 5))                
                x,y,z = pos[0], pos[1], pos[2]
                file2.write(str(self.now - self.start_exe) + ", "  + str(x) + ', ' + str(y) + ', ' + str(z))
                file2.write('\n')
                file2.close()
        #Check if all waypoints have been passed through
        if self.current_waypoint == len(self.waypoints):
            self.current_waypoint = 0
            self.next_state = self.previous_state
            print("Next State Equals")
            print(self.next_state)
            print('\n')
        cmd_time = time.time()-self.zTime
        self.status_message = "State: Execute - Executing motion plan, Err = %0.2f Time since cmd: %0.2f MaxTime: %0.2f" %(self.err, cmd_time, self.rxarm.moving_time)  

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

        print(self.camera.extrinsic_matrix)
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
        self.current_waypoint = 0
        self.err = 0
        self.next_state = "idle"

    def save_trajectory(self):
        """!
        @brief      Export set of waypoints currently in memory to a .txt file
        """        
        self.current_state = "save_waypoints"
        self.status_message = "Saving waypoints"
        #Record data
        x = []
        y = []
        z = []
        file1 = open("Data.txt","a")
        
        file1.write('New Data')
        file1.write('\n')
        file1.write('\n')
        
        for wp in self.waypoints:
            pos = get_pose_from_T(FK_dh(self.rxarm.dh_params, wp, 5))
            x = pos[0]
            y = pos[1]
            z = pos[2]        
            print(str(x) + ', ' + str(y) + ', ' + str(z))            
            file1.write(str(x) + ', ' + str(y) + ', ' + str(z))
            file1.write('\n')
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

            print(self.camera.homography)

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
            
            self.camera.new_click = False
            self.next_state = "execute"

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

    def event1(self):
        """!
        @brief      Mix of small and large blocks, may be stacked. Goal is to move all small blocks to the right and all big blocks to the right of the arm
        """

        #This code framework does not currently account for small blocks hidden under large blocks, this can maybe be accounted for by separating blockredetection
        #from normal block detection and checking if there is a significant difference in the countour width and height
        # Can also try checking this by seeing if any of the heights are reasonable multiples of small blocks(n*25) or of big blocks (n*37.5)

        self.status_message = "Running Event 1"
        self.current_state = "event1"
        
        self.waypoints = []
        self.waypoint_grip = []

        if self.init1:
            self.smallBlockPlace1 = np.zeros((4,4))
            self.smallBlockPlace1[0:3,0:3] = get_R_from_euler_angles(0.0,np.pi,0.0)
            self.bigBlockPlace1 = self.smallBlockPlace1.copy()
            self.smallBlockPlace1[:,-1] = 50*np.array([-2.5, -2.5, 0.25, 1/50])
            self.bigBlockPlace1[:,-1] = 50*np.array([2.5, -2.5, 0, 1/50])
            self.init1 = False

        self.robot_home()
        #self.rxarm.sleep() 
        #rospy.sleep(3)
        #Ensure that block detection only happens once arm is fully slept
        self.detect_blocks_once()
        
        moves = 0

        for block in self.camera.blockData:
            block_Frame = block[0]
            theta = block[2]
            x = block_Frame[0,3]
            y = block_Frame[1,3]
            block_type = block[5]

            # print(block_Frame[:,-1])
            # print(block_type)
            # print(theta)
            # print('')

            self.thisBlock = [x,y,block_type]
            #Move small blocks to negative block area
            if y>=0 and block_type == 'Small Block':
                self.pick_up_block(block_Frame,theta,12)
                self.place_block(self.smallBlockPlace1,"down",3)
                self.smallBlockPlace1[0,-1] -= 50
                moves += 1
            #Move big blocks to positive block araea
            elif y>=0 and block_type == 'Big Block':
                self.pick_up_block(block_Frame,theta, 15)
                if self.bigBlockPlace1[0,-1] < 350 :
                    self.bigBlockPlace1[0,-1] += 50
                else:
                    self.bigBlockPlace1[1,-1] += 70
                    self.bigBlockPlace1[0,-1] = 50*2.5

                self.place_block(self.bigBlockPlace1,"down",25)
                moves +=1
        #Done if robot determined that there was nothin to move
        if moves == 0:
            self.next_state = 'idle'
        else:
            self.next_state = 'executeAndReturn'            

    def event2(self):
        """!
        @brief      Sort small blocks to left in stacks of 3 and large blocks to the right in stacks of 3
        """
        self.status_message = "Running Event 2"
        self.current_state = "event2"

        self.waypoints = []
        self.waypoint_grip = []

        #This code framework does not currently account for small blocks hidden under large blocks, this can maybe be accounted for by separating blockredetection
        #from normal block detection and checking if there is a significant difference in the countour width and height
        # Can also try checking this by seeing if any of the heights are reasonable multiples of small blocks(n*25) or of big blocks (n*37.5)        

        if self.init2:
            self.smallBlockPlace2 = np.zeros((4,4))
            self.smallBlockPlace2[0:3,0:3] = get_R_from_euler_angles(0.0,np.pi,0.0)
            self.bigBlockPlace2 = self.smallBlockPlace2.copy()
            self.smallBlockPlace2[:,-1] = 50*np.array([-4, -2.5, 0.25, 1/50])
            self.bigBlockPlace2[:,-1] = 50*np.array([4, -2.5, 0, 1/50])
            self.init2 = False

        self.robot_home() 
        # rospy.sleep(3)
        #Ensure that block detection only happens once arm is fully slept
        self.detect_blocks_once()
        
        moves = 0
        for block in self.camera.blockData:
            block_Frame = block[0]
            theta = block[2]
            x = block_Frame[0,3]
            y = block_Frame[1,3]
            block_type = block[5]

            # print(block_Frame[:,-1])
            # print(block_type)
            # print(theta)
            # print('')
            self.thisBlock = [x,y,block_type]
            #Move small blocks to negative block area
            if y>=0 and block_type == 'Small Block':
                self.pick_up_block(block_Frame,theta,17)
                self.place_block(self.smallBlockPlace2,"down",3)
                if self.smallStackCt2 < 2:
                    self.smallBlockPlace2[2, -1] += 23
                else:
                    self.smallBlockPlace2[2, -1] = 0
                    self.smallBlockPlace2[0,-1] -= 50
                    self.smallStackCt2 = -1

                self.smallStackCt2 += 1
                moves += 1
            #Move big blocks to positive block araea
            elif y>=0 and block_type == 'Big Block':
                self.pick_up_block(block_Frame,theta, 15)
                self.place_block(self.bigBlockPlace2,"down",25)
                if self.bigStackCt2 < 2:
                    self.bigBlockPlace2[2, -1] += 25
                else:
                    self.bigBlockPlace2[2, -1] = 0
                    self.bigBlockPlace2[0,-1] += 75
                    self.bigStackCt2 = -1

                self.bigStackCt2 += 1

                moves +=1
        #Done if robot determined that there was nothin to move
        if moves == 0:
            self.next_state = 'idle'
        else:
            self.next_state = 'executeAndReturn'          

    def event3(self):
        """!
        @brief      Line up in ROYGBV order, separate lines for small and large blocks, line can be at most 30cm (12 small blocks or 8 large blocks)
        """
        self.status_message = "Running Event 3"
        self.current_state = "event3"

        #This code framework does not currently account for small blocks hidden under large blocks, this can maybe be accounted for by separating blockredetection
        #from normal block detection and checking if there is a significant difference in the countour width and height
        # Can also try checking this by seeing if any of the heights are reasonable multiples of small blocks(n*25) or of big blocks (n*37.5)        

        self.waypoints = []
        self.waypoint_grip = []

        if self.init3:
            self.smallBlockPlace3 = np.zeros((4,4))
            self.smallBlockPlace3[0:3,0:3] = get_R_from_euler_angles(0.0,np.pi,0.0)
            self.bigBlockPlace3 = self.smallBlockPlace3.copy()
            self.smallBlockPlace3[:,-1] = 50*np.array([-1.25, -2.5, 0.25, 1/50])
            self.bigBlockPlace3[:,-1] = 50*np.array([1.25, -2.5, 0, 1/50])
            self.init3 = False

        self.robot_home() 
        rospy.sleep(3)
        #Ensure that block detection only happens once arm is fully slept
        self.detect_blocks_once()
        
        moves = 0
        #Sort by color so that the blocks end up being added to waypoints in color order => blocks get placed in color order
        self.pre_process3()
        #self.rxarm.set_moving_time(1.4)

        for block in self.camera.blockData:
            block_Frame = block[0]
            theta = block[2]
            x = block_Frame[0,3]
            y = block_Frame[1,3]
            block_type = block[5]
            color = block[1]
            print('EVENT3 STUFF')


            print(x)
            print(y)
            print(block_type)
            print(theta)
            print(color)
            print('')
            self.thisBlock = [x,y,block_type]
            try:
                #Move small blocks to appropriate color position
                if y>=0 and block_type == 'Small Block':
                    self.pick_up_block(block_Frame,theta,17)
                    place_color = self.smallBlockPlace3.copy()
                    if self.Three2Four:
                        place_color[0,-1] = 50*self.event4smallX[color]
                        place_color[1,-1] = 50*self.event4smallY[color]
                    else:
                        place_color[0,-1] = 50*self.event3small[color]
                    #print(50*self.event3small[color])
                    self.place_block(place_color,"down",3)
                    moves += 1
                #Move big blocks to positive block araea
                elif y>=0 and block_type == 'Big Block':
                    self.pick_up_block(block_Frame,theta, 15)
                    place_color = self.bigBlockPlace3.copy()
                    if self.Three2Four:
                        place_color[0,-1] = 50*self.event4bigX[color]
                        place_color[1,-1] = 50*self.event4bigY[color]
                    else:
                        place_color[0,-1] = 50*self.event3big[color]                    
                    self.place_block(place_color,"down",25)
                    moves +=1
            except:
                print("SKIP THIS BLOCK")
        #Done if robot determined that there was nothin to move
        if moves == 0:
            #If called from event4(), return to event4()
            if self.Three2Four:
                self.next_state = 'event4'
                self.Three2Four = False
            else:
                self.next_state = 'idle'
        else:
            self.next_state = 'executeAndReturn'            

    def event4(self):
        """!
        @brief      Stack up in ROYGBV order, separate lines for small and large blocks
        """
        self.status_message = "Running Event 4"
        self.current_state = "event4"

        #This code framework does not currently account for small blocks hidden under large blocks, this can maybe be accounted for by separating blockredetection
        #from normal block detection and checking if there is a significant difference in the countour width and height
        # Can also try checking this by seeing if any of the heights are reasonable multiples of small blocks(n*25) or of big blocks (n*37.5)        
        self.waypoints = []
        self.waypoint_grip = []

        if self.init4:
            #Run event3 first so that we have an easy set of blocks to work with
            self.smallBlockPlace4 = np.zeros((4,4))
            self.smallBlockPlace4[0:3,0:3] = get_R_from_euler_angles(0.0,np.pi,0.0)
            self.bigBlockPlace4 = self.smallBlockPlace4.copy()
            self.smallBlockPlace4[:,-1] = 50*np.array([-6, 3.5, 0, 1/50])
            self.bigBlockPlace4[:,-1] = 50*np.array([6, 3.5, 0, 1/50])
            self.init4 = False

            self.Three2Four = True
            self.next_state = "event3"
            print('Go to event3')

        else:
            self.robot_home() 
            rospy.sleep(3)
            #Ensure that block detection only happens once arm is fully slept
            self.detect_blocks_once()
            
            moves = 0

            #Sort by color so that it moves blocks in order of color
            self.camera.blockData.sort(key = self.color_sort)
            for block in self.camera.blockData:
                block_Frame = block[0]
                theta = block[2]
                if theta <80:
                    theta += 90

                x = block_Frame[0,3]
                y = block_Frame[1,3]
                block_type = block[5]

                print(block_Frame[:,-1])
                print(block_type)
                print(theta)
                print('')
                self.thisBlock = [x,y,block_type]

                #Move small blocks to negative block area
                if y<=1 and block_type == 'Small Block':
                    self.pick_up_block(block_Frame,theta,5)
                    self.place_block(self.smallBlockPlace4,"down",3)
                    self.smallBlockPlace4[2, -1] += 25
                    moves += 1
                #Move big blocks to positive block araea
                elif y<=1 and block_type == 'Big Block':
                    self.pick_up_block(block_Frame,theta, 10)
                    self.place_block(self.bigBlockPlace4,"down",25)
                    self.bigBlockPlace4[2, -1] += 35

                    moves +=1
            #Done if robot determined that there was nothin to move
            if moves == 0:
                self.next_state = 'idle'
            else:
                self.next_state = 'executeAndReturn'   

    def bonus(self):
        print("STARTING BONUS")
        self.current_state = 'bonus'

        'Click on GUI to exit bonus'
        if self.camera.new_click:
            print(self.bonus_height)
            self.next_state = 'idle'

        if self.init_b:
            self.now_b = time.time()

            self.detect_blocks_once()
            self.bonus_stack_loc = self.camera.blockData[0][0]
            self.bonus_stack_loc[2,-1] = 17 # Pick up from middle
            self.bonus_theta = self.camera.blockData[0][2]
            self.init_b = False
            self.bonus_height = 1

        #We get dur seconds to place a new block between each place
        dur = 5
        if (time.time() - self.now_b >dur):
            self.detect_blocks_once()

            #For first block to add, make sure the color comes before initial block (ROYGBV)
            if self.bonus_height == 1:
                self.camera.blockData.sort(key = self.color_sort)
            else:
                self.camera.blockData.sort(key = self.height_sort)

            self.place_stack_on_block = self.camera.blockData[0][0]

            #I dont know what the direction name should be, but this should come from the side
            self.pick_up_block(self.bonus_stack_loc,theta,0, bonus_ = True)
            self.place_block(self.place_stack_on_block, "down",33, bonus_ = True)

            self.bonus_height+=1
            #Can optionally just try to put an intelligent mask here
            self.rxarm.sleep()
            rospy.sleep(2)
        else:
            remain_t = dur - (time.time() - self.now_b)
            self.status_message =  "Time left to place new block: %0.2f "%remain_t
    
    def bonusRob(self):

        print("STARTING BONUS")
        self.rxarm.moving_time = 0.5
        self.rxarm.accel_time = 0.2
        self.current_state = 'bonus'
        self.waypoints = []
        self.waypoint_grip = []
        extraWrist = 5 # + self.sizeTower**2/20 #Keeps arm stable with heavier loads
        extraShoulder = 0#self.sizeTower/10
        one =   [-43, 35.6, -44.9, 80.6, 0.0]

        two =   [-43, 21.3-extraShoulder, -50, 69.2+extraWrist, 0.0]

        three = [-43, 26.5-extraShoulder, -34.8, 61.3+extraWrist, 0.0]

        laying =  [-43, 27.85, -34.6, 62.5+extraWrist, 0.0]

        four =  [-43, 38.7, -32.1, 70.7+extraWrist, 0.0]

        print(self.isGrabbingInFront)

        for i,v in enumerate(one):
            one[i] *= math.pi/180
            two[i] *= math.pi/180
            three[i] *= math.pi/180
            four[i] *= math.pi/180
            laying[i] *=math.pi/180


        time.sleep(0.5)
        (x * 2 for x in [2, 2])
        # 'Click on GUI to exit bonus'
        if self.isGrabbingInFront:
            if self.sizeTower == 1:
                self.waypoints.append(one)
                self.waypoint_grip.append(0)

                self.waypoints.append(one)
                self.waypoint_grip.append(1)

            self.waypoints.append(two)
            self.waypoint_grip.append(1)

            self.waypoints.append(three)
            self.waypoint_grip.append(1)

            self.waypoints.append(laying)
            self.waypoint_grip.append(1)

            self.waypoints.append(laying)
            self.waypoint_grip.append(0.05)

            self.waypoints.append(four)
            self.waypoint_grip.append(0.05)

            self.waypoints.append(four)
            self.waypoint_grip.append(0)

            self.waypoints.append(four)
            self.waypoint_grip.append(1)

            self.waypoints.append(one)
            self.waypoint_grip.append(1)

            
        self.sizeTower += 1
        self.current_waypoint = 0
        self.next_state = "executeAndReturn"
        

#Doesn't work since we can't install mpl_toolkits
    def plot(self):
        self.current_state = "plot"
        self.next_state = "idle"
        x = []
        y = []
        z = []
        for wp in self.waypoints:
            pos = get_pose_from_T(FK_dh(self.rxarm.dh_params, wp, 5))
            print(pos[0:3])
            x += pos[0]
            y += pos[1]
            z += pos[2]
            
        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.plot3D(x, y, z)
        ax.xlabel("x [mm]")
        ax.ylabel("y [mm]")
        ax.zlabel("z [mm]")
        ax.title('Teach and Repeat EE Trajectory')

        plt.show()                
#--------------------------Helper Functions-------------------------------------------------#
    def actuate_gripper(self,grip_state):
        #Actuate Grippers
        if grip_state:
            self.rxarm.close_gripper()
        else:
            self.rxarm.open_gripper()
    
    def detect_blocks_once(self):
        self.camera.detectBlocksInDepthImage()
        self.camera.blockDetector()

    def pick_up_block(self, pickupFrame, theta, offset, bonus_ = False):
        """!
        @brief      Adds waypoints to pick up block
        """        
        # self.waypoints = []
        # self.waypoint_grip = []

        approachFrame = pickupFrame.copy()
        approachFrame[2,-1] += 150
        if bonus_:
            approachFrame[2,-1] = 100        
        radius = np.linalg.norm(pickupFrame[0:1,-1]) # Stopgap solution- this increases the offset as the block is farther away to correct for gravity
        pickupFrame[2,-1] += radius/20

        #TODO: write IK function to accomplish the rest of this
        approach = IK_geometric_event_1(self.rxarm.dh_params, approachFrame,theta) 
        # print(get_pose_from_T(FK_dh(self.rxarm.dh_params, approach, 5)))
        # print('')
        print(self.thisBlock)
        pickup = IK_geometric_event_1(self.rxarm.dh_params, pickupFrame,theta)        
        # print(get_pose_from_T(FK_dh(self.rxarm.dh_params, pickup, 5)))
        # print('')
        self.waypoints.append(approach)
        self.waypoint_grip.append(0)

        self.waypoints.append(pickup)
        self.waypoint_grip.append(0)

        self.waypoints.append(pickup)
        self.waypoint_grip.append(1)

        self.waypoints.append(approach)
        self.waypoint_grip.append(1)

        # self.waypoints.append([0,0,0,0,0])
        # self.waypoint_grip.append(1)

    def place_block(self, placeFrame, direction, offset, bonus_ = False):
        """!
        @brief      Adds waypoints to place block
        """                
        # self.waypoints = []
        # self.waypoint_grip = []
        approachFrame = placeFrame.copy()
        approachFrame[2,-1] += 150
        if bonus_:
            approachFrame[2,-1] = 100
        placeFrame[2,-1] += offset

        #TODO: write IK function to accomplish the rest of this   
        approach = IK_geometric_event_1(self.rxarm.dh_params, approachFrame,90)    
        # print(get_pose_from_T(FK_dh(self.rxarm.dh_params, approach, 5)))
        # print('')
        drop = IK_geometric_event_1(self.rxarm.dh_params, placeFrame,90)
        # print("Block Angle Desired:")
        # print(drop[-1]-drop[0])
        # print("\n")

        # print(get_pose_from_T(FK_dh(self.rxarm.dh_params, drop, 5)))
        # print('')
        
        self.waypoints.append(approach)
        self.waypoint_grip.append(1)

        self.waypoints.append(drop)
        self.waypoint_grip.append(1)

        self.waypoints.append(drop)
        self.waypoint_grip.append(0)

        self.waypoints.append(approach)
        self.waypoint_grip.append(0)       

        self.waypoints.append(approach)
        self.waypoint_grip.append(0)

        # self.waypoints.append([0,0,0,0,0])
        # self.waypoint_grip.append(0)     

    def next_color(self, current_color):
        #Determines next color in ROYGBV
        current_color_idx = self.ROYGBV_idx[current_color]
        return ROYGBV[(current_color_idx+1)%6 ]

    def color_sort(self, blockData):
        return self.ROYGBV_idx[blockData[1]]

    def height_sort(self,blockData):
        return(blockData[0][2,-1])

    def robot_home(self, wait = 2):
        self.rxarm.set_positions([0, -np.pi/2+np.pi/12, -np.pi/2, np.pi/2, 0])          
        self.rxarm.open_gripper()
        rospy.sleep(wait)

    def pre_process3(self):
        smallblocks = []
        bigblocks = []
        for blockData in self.camera.blockData:
            if blockData[5] == 'Small Block':
                smallblocks.append(blockData)
            elif blockData[5] == 'Big Block':
                bigblocks.append(blockData)
        
        bigblocks.sort(key = self.color_sort, reverse = True) #VBGYOR 
        smallblocks.sort(key = self.color_sort) #ROYGBV
        self.camera.blockData = bigblocks+smallblocks

#Testing color_sort
    def test(self):
        # print('wut')
        self.next_state = 'idle'
        self.current_state = 'test'
        self.detect_blocks_once()

        for data in self.camera.blockData:
            print(data[1])

        # print('')
        # print('Sorting')
        # print('')
        self.camera.blockData.sort(key=self.color_sort)

        for data in self.camera.blockData:
            print(data[1])
#Testing self.camera.PxFrame2WorldFrame()
'''
    def test(self):
        self.detect_blocks_once()
        self.pick_up_block(self.camera.blockData[0][0],"down")
        bigBlockPlace1 = self.camera.blockData[0][0].copy()
        bigBlockPlace1[0][-1] -= 250
        self.place_block(bigBlockPlace1,"down")   
        self.next_state = "executeAndReturn"
'''
#Testing block_in_zone
'''
    def test(self):
        UR = 50*np.array([0,8.5,0])
        LL = 50*np.array([4,4.5,0])
        zone = [UR, LL]
        BIZ = self.camera.block_in_zone(zone)

        if BIZ:
            self.detect_blocks_once()
            blocks_to_move = []
            for block in BIZ:
                blocks_to_move.append(self.camera.contour_id(block))            
            print(blocks_to_move)
            next_state = 'idle'
'''


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



        # #Ensure that there are no blocks in negative zone
        # UL = 50*np.array([9,1.5,0])
        # LR = 50*np.array([4,-2.5,0])
        # zone = [UL, LR]
        # BIZ = self.camera.block_in_zone(zone)
        # if BIZ:
        #     blocks_to_move = []
        #     for block in BIZ:
        #         blocks_to_move.append(self.camera.contour_id(block))
        # #blocks_to_move now contains blockData of blocks that need to be moved from the zone
        # #TODO: MOVE BLOCKS

        # #Ensure that there are no blocks in positive zone
        # UL = 50*np.array([-4,1.5,0])
        # LR = 50*np.array([-9,-2.5,0])
        # zone = [UL, LR]
        # BIZ = self.camera.block_in_zone(zone)
        # if BIZ:
        #     blocks_to_move = []
        #     for block in BIZ:
        #         blocks_to_move.append(self.camera.contour_id(block))
        # #TODO: MOVE BLOCKS            