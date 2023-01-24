"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm
import math


def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    pass


def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix T from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transformation matrix.
    """
    T = np.zeros((4,4))
    T = np.array([[np.cos(theta),   -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],\
                    [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],\
                    [0,              np.sin(alpha),                np.cos(alpha),               d], \
                    [0,              0,                            0,                           1] ])
    return T

def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the 3 Euler angles from a 4x4 transformation matrix T
                If you like, add an argument to specify the Euler angles used (xyx, zyz, etc.)

    @param      T     transformation matrix

    @return     The euler angles from T. Euler angles used ZYZ
    """
    R = T[0:3,0:3]  #Rotation Matrix

    theta = math.atan2(math.sqrt(1-R[2][2]**2),R[2][2])
    if(math.sin(theta)>0):
        phi = math.atan2(R[1][2],R[0][2])
        psi = math.atan2(R[2][1],-R[2][0])
    else:
        phi = math.atan2(-R[1][2],-R[0][2])
        psi = math.atan2(-R[2][1],R[2][0])
    return (phi,theta,psi)


def get_R_from_euler_angles(phi,theta,psi):
    """!
    @brief      Gets the rotation matrix (ZYZ) from euler angles.

    @param      Euler angles phi, theta, psi 

    @return     Rotation matrix (ZYZ)
    """
    R = np.eye(3)
    c1 = np.cos(phi)
    s1 = np.sin(phi)
    c2 = np.cos(theta)
    s2 = np.sin(theta)
    c3 = np.cos(psi)
    s3 = np.sin(psi)
    R = np.array([[c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3, c1*s2],
                         [c1*s3+c2*c3*s1, c1*c3-c2*s1*s3, s1*s2],
                         [-c3*s2, s2*s3, c2]])
    return R



def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the 6DOF pose vector from a 4x4 transformation matrix T

    @param      T     transformation matrix

    @return     The pose vector from T.
    """
    phi,theta,psi = get_euler_angles_from_T(T)

    return [T[0][3], T[1][3], T[2][3], phi, theta, psi]

def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a  representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4x4 homogeneous matrix representing the pose of the desired link

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4x4 homogeneous matrix representing the pose of the desired link
    """
    # no_joint_angles = max(np.size(joint_angles, 0),np.size(joint_angles, 1))
    # s_1_mat = to_s_matrix(s_lst[0:3,0],s_lst[3:6,0])
    # s_2_mat = to_s_matrix(s_lst[0:3,1],s_lst[3:6,1])
    # s_3_mat = to_s_matrix(s_lst[0:3,2],s_lst[3:6,2])

    # R_1_mat = expm(s_1_mat*joint_angles[0])
    # R_2_mat = expm(s_2_mat*joint_angles[1])
    # R_3_mat = expm(s_2_mat*joint_angles[2])

    # return R_1_mat.dot(R_2_mat).dot(R_3_mat).dot(m_mat)
    pass



def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    # skew_w = skew(w)
    # v_array = np.vstack(v)
    # z_array = np.zeros((1, 4))
    # S_matrix = np.c_(skew_w,v_array)
    # return np.r_(S_matrix,z_array)
    pass
    

def IK_geometric(dh_params, pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array 

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    pass