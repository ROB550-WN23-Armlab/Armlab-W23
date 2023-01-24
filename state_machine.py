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
        self.current_waypoint = -1
        self.waypoint_grip = [0,0,0,0,0,0,0,0,0,0] #Read 1 as closed and 0 as open
        self.err = 0
        self.thresh = 0.05
        self.long_time = 0
        self.record = 0

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
            self.execute()

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
        self.status_message = "State: Idle - Waiting for input hi"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """

        self.current_state="execute"

        if self.current_waypoint == -1:
            self.now = time.time()
            self.start_exe = time.time()

        
        #Actuate Grippers
        if self.waypoint_grip[self.current_waypoint]:
            self.rxarm.close_gripper()
        else:
            self.rxarm.open_gripper()
        
         #Calculate Joint Errors
        joint_errors = np.array(self.waypoints[self.current_waypoint]) - np.array(self.rxarm.get_positions())
        self.err = np.sqrt(np.mean(joint_errors**2))

        #Waypoint checking and actuation
        if (self.err < self.thresh) or (self.current_waypoint < 0) or (self.long_time):
            self.current_waypoint = self.current_waypoint + 1
            self.rxarm.set_positions(self.waypoints[self.current_waypoint])          
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
        if self.current_waypoint == len(self.waypoints)-1:
            self.current_waypoint = -1
            self.next_state = "idle"
        self.status_message = "State: Execute - Executing motion plan, Err =" + str(self.err) + "Time since cmd:" + str(time.time()-self.zTime)


    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        image_points = np.zeros((8,3))
        model_points = np.zeros((8,3))

        """TODO Perform camera calibration routine here"""
        for idx, tag in enumerate(self.camera.tag_detections.detections):
            x = tag.pose.pose.pose.position.x
            y = tag.pose.pose.pose.position.y
            z = tag.pose.pose.pose.position.z
            image_points[idx,:] = (np.matmul(self.camera.intrinsic_matrix,np.array([x,y,z]).reshape(3,1))/z).reshape(3)
            model_points[idx,:] = np.array(self.camera.tag_locations[tag.id[0]-1])
        image_points=image_points[:,0:2].astype('float32')
        model_points=model_points.astype('float32')
        image_points = image_points[~np.all(image_points == 0, axis=1)]
        model_points = model_points[~np.all(model_points == 0, axis=1)]
        print(image_points)
        print('\n')
        print(model_points)
        print('\n\n\n')
        (success,rot_vec,trans_vec) = cv2.solvePnP(model_points,
                                                   image_points,
                                                   self.camera.intrinsic_matrix,
                                                   self.camera.dist_coeffs,
                                                   flags=cv2.SOLVEPNP_ITERATIVE)

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

        spatial_transform = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]], dtype=np.float64)

        spatial_transform[0:3,3] = trans_vec.reshape(3)
        spatial_transform[0:3,0:3] = rotation_mat

        self.camera.extrinsic_matrix = np.linalg.inv(spatial_transform)

        self.status_message = "Calibration - Completed Calibration"

    """ TODO """
    def detect(self):
        """!
        @brief      Detect the blocks
        """
        rospy.sleep(1)

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