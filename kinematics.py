"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm
from numpy import linalg as la
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
    T_mat = np.eye(4)
    joint_angles_mod = [joint_angles[0]+math.pi/2,
                        -joint_angles[1]+math.atan2(200,50),
                        joint_angles[2]-math.atan2(200,50),
                        joint_angles[3]+math.pi/2,
                        joint_angles[4]]

    #print(joint_angles_mod)
    for i in range(5):
        dh_params[i][-1] = joint_angles_mod[i]

    for j in range(link):
        a = dh_params[j][0]
        alpha = dh_params[j][1]
        d = dh_params[j][2]
        theta = dh_params[j][3] 

        #print(get_transform_from_dh(a,alpha,d,theta))
        T_mat = np.matmul(T_mat,get_transform_from_dh(a,alpha,d,theta))
    
    #print(T_mat.astype('int'))
    #print('\n')


    return T_mat

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
    T = np.array((4,4))
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

    # try:
    theta = math.atan2(math.sqrt(1-(R[2][2])**2),R[2][2])
    # print(R)
    # print('\n')
    # print(theta)
    # print('\n')
    if(math.sin(theta)>0):
        phi = math.atan2(R[1][2],R[0][2])
        psi = math.atan2(R[2][1],-R[2][0])
    else:
        phi = math.atan2(-R[1][2],-R[0][2])
        psi = math.atan2(-R[2][1],R[2][0])
    return (phi,theta,psi)
    # except:
    #     print(R)
    #     print('\n')
    #     print(R[2][2])
    #     print('\n')

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
    l1 = 205.73
    l2 = 200
    d1 = 103.91
    angle_offset = math.pi/2 - math.atan2(50,200)
    phi = pose[3]
    theta = pose[4]
    psi = pose[5]
    
    R_0_5 = get_R_from_euler_angles(phi,theta,psi)

    # End Effector Location
    pos_ee = np.array([[pose[0]],[pose[1]],[pose[2]]])
    x = pos_ee[0]
    y = pos_ee[1]
    z = pos_ee[2]

    link6_len = dh_params[4][2]; 
    pos_wrist = pos_ee - link6_len*np.matmul(R_0_5,np.array([[0],[0],[1]]))

    # Wrist location
    ox = pos_wrist[0]
    oy = pos_wrist[1]
    oz = pos_wrist[2]

    planar_x = math.sqrt(ox**2 + oy**2)
    planar_y = oz - d1
    # Q1 - Calculation 
    q1 = math.atan2(oy,ox) - np.pi/2.0
    if q1 < -np.pi:
        q1 = q1 + 2*np.pi

    # Q3 - Calculation
    theta_3 = math.acos(((planar_x**2 + planar_y**2) - l1**2 -l2**2)/(2*l1*l2))
    theta_3 = -theta_3    # Choosing Elbow Up 
    q3 = theta_3 + angle_offset

    # Q2 - Calculation
    theta_2 = math.atan2(planar_y,planar_x) - math.atan2(l2*math.sin(theta_3),l1+l2*math.cos(theta_3))
    q2 = angle_offset - theta_2

    ## For Q3, Q4, Q5 calculations
    joint_angles_mod = np.array([q1,q2,q3,0.0,0.0])
    T_0_3 = FK_dh(dh_params, joint_angles_mod, 3)
    R_0_3 = T_0_3[0:3,0:3]
    R_3_5 = np.matmul(np.linalg.inv(R_0_3), R_0_5)

    q4,q5,q6 = get_euler_angles_from_T(R_3_5)

    return [q1,q2,q3,q4,q6-math.pi/2]

'''
def IK_geometric_two(dh_params, pose, direction):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array 
    @param      direction  The direction of EE, it could be "flat" or "down"

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    l1 = 205.73
    l2 = 200
    d1 = 103.91
    angle_offset = math.pi/2 - math.atan2(50,200)
    # phi = pose[3]
    # theta = pose[4]
    # psi = pose[5]

    # End Effector Location
    pos_ee = np.array([pose[0],pose[1],pose[2]])
    x = pos_ee[0]
    y = pos_ee[1]
    z = pos_ee[2]


    link6_len = dh_params[4][2]; 
    if direction=="down":
        pos_wrist = pos_ee + np.array([0,0,link6_len])
    elif direction=="flat":
        # R_0_5 = rotation_matrix(-90,-90,0,'xzy')
        pos_wrist = pos_ee - np.transpose(np.append( pos_ee[0:2]*link6_len/la.norm(pos_ee[0:2]),0))
        print([pos_ee])

    # R_0_5 = get_R_from_euler_angles(phi,theta,psi)

    
    
    link6_len = dh_params[4][2]; 
    # pos_wrist = pos_ee - link6_len*np.matmul(R_0_5,np.array([[0],[0],[1]]))

    # Wrist location
    ox = pos_wrist[0]
    oy = pos_wrist[1]
    oz = pos_wrist[2]
    
    planar_x = math.sqrt(ox**2 + oy**2)
    planar_y = oz - d1
    # Q1 - Calculation 
    q1 = math.atan2(oy,ox) - np.pi/2.0
    if q1 < -np.pi:
        q1 = q1 + 2*np.pi

    # Q3 - Calculation
    print((((planar_x**2 + planar_y**2) - l1**2 -l2**2)/(2*l1*l2)))

    try:
        theta_3 = math.acos(((planar_x**2 + planar_y**2) - l1**2 -l2**2)/(2*l1*l2))
    except:
        theta_3 = 0
        print('error in theta_3')

    theta_3 = -theta_3    # Choosing Elbow Up 
    q3 = theta_3 + angle_offset

    # Q2 - Calculation
    theta_2 = math.atan2(planar_y,planar_x) - math.atan2(l2*math.sin(theta_3),l1+l2*math.cos(theta_3))
    q2 = angle_offset - theta_2

    ## For Q3, Q4, Q5 calculations
    # joint_angles_mod = np.array([q1,q2,q3,0.0,0.0])
    # T_0_3 = FK_dh(dh_params, joint_angles_mod, 3)
    # R_0_3 = T_0_3[0:3,0:3]
    # R_3_5 = np.matmul(np.linalg.inv(R_0_3), R_0_5)

    # q4,q5,q6 = get_euler_angles_from_T(R_3_5)

    if direction=="down":
        q4 = q2-q3-math.pi/2  
        q5 = q1
    elif direction=="flat":
        q4 = q2-q3
        q5 = 0

    return [q1,q2,q3,q4,q5]
'''

def IK_geometric_two(dh_params, pose, direction):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array 
    @param      direction  The direction of EE, it could be "flat" or "down"

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    # Defining co
    l1 = dh_params[1][0] # 205.73
    l2 = dh_params[2][0] # 200
    d1 = dh_params[0][2] # 103.91
    link6_len = dh_params[4][2]; 
    angle_offset = math.pi/2 - math.atan2(50,200)
    # print(l1,l2,d1,link6_len,angle_offset)
    # print('\n')

    # End Effector Location
    pos_ee = np.array([pose[0],pose[1],pose[2]])
    x = pos_ee[0]
    y = pos_ee[1]
    z = pos_ee[2]
    # print(pos_ee)
    # print('\n')

    if direction=="down":
        pos_wrist = pos_ee + np.array([0,0,link6_len])
        # print(pos_wrist)
        # print('\n')
        # print(link6_len)
        # print('\n')
        R_0_5 = get_R_from_euler_angles(0.0,np.pi,0.0)
    elif direction=="flat":
        R_0_5 = get_R_from_euler_angles(math.atan2(pose[1],pose[0]),np.pi/2,0.0)
        pos_wrist = pos_ee - link6_len*np.matmul(R_0_5,np.array([0,0,1]))

    # Wrist location
    ox = pos_wrist[0]
    oy = pos_wrist[1]
    oz = pos_wrist[2]
    # print(ox,oy,oz)
    # print('\n')
    planar_x = math.sqrt(ox**2 + oy**2)
    planar_y = oz - d1
    # print(planar_x)
    # print('\n')
    # print(planar_y)
    # print('\n')
    # Q1 - Calculation 
    q1 = math.atan2(oy,ox) - np.pi/2.0
    if q1 < -np.pi:
        q1 = q1 + 2*np.pi

    # Q3 - Calculation
    c3 = ((planar_x**2 + planar_y**2) - l1**2 -l2**2)/(2*l1*l2)
    
    if np.absolute(c3) > 1:
        print('Postion Unreachable')
    elif c3 ==1:
        theta_2 = math.atan2(oy,ox)
        theta_3 = 0
        theta_3_alt = 0
    elif c3 ==-1 and (planar_x**2 + planar_y**2) != 0:
        theta_2 = math.atan2(oy,ox)
        theta_3 = math.pi
        theta_3_alt = -math.pi
    elif c3 ==-1 and (planar_x**2 + planar_y**2) == 0:
        theta_2 = 0 # picking zero, as it has infinite solutions
        theta_3 = math.pi
        theta_3_alt = -math.pi
        print('Infinite solutions')
    else:
        theta_3 = math.acos(c3)
        theta_3_alt = -math.acos(c3)

    theta_3 = -theta_3    # Choosing Elbow Up 
    q3 = theta_3 + angle_offset
    q3_alt = theta_3_alt + angle_offset

    # Q2 - Calculation
    theta_2 = math.atan2(planar_y,planar_x) - math.atan2(l2*math.sin(theta_3),l1+l2*math.cos(theta_3))
    theta_2_alt = math.atan2(planar_y,planar_x) - math.atan2(l2*math.sin(theta_3_alt),l1+l2*math.cos(theta_3_alt))
    q2 = angle_offset - theta_2
    q2_alt = angle_offset - theta_2_alt

    q_1_2_3_A1 = np.array([q1,q2,q3])
    q_1_2_3_A2 = np.array([q1,q2_alt,q3_alt])

    # For A1 - For Q3, Q4, Q5 calculations
    joint_angles_mod_A1 = np.array([q1,q2,q3,0.0,0.0])
    T_0_3_A1 = FK_dh(dh_params, joint_angles_mod_A1, 3)
    R_0_3_A1 = T_0_3_A1[0:3,0:3]
    R_3_5_A1 = np.matmul(np.linalg.inv(R_0_3_A1), R_0_5)

    # For A2 - For Q3, Q4, Q5 calculations
    joint_angles_mod_A2 = np.array([q1,q2_alt,q3_alt,0.0,0.0])
    T_0_3_A2 = FK_dh(dh_params, joint_angles_mod_A2, 3)
    R_0_3_A2 = T_0_3_A2[0:3,0:3]
    R_3_5_A2 = np.matmul(np.linalg.inv(R_0_3_A2), R_0_5)

    q4_A1,q5_A1,q6_A1 = get_euler_angles_from_T(R_3_5_A1)
    q4_A2,q5_A2,q6_A2 = get_euler_angles_from_T(R_3_5_A2)

    # Checking q1,q2,q3,q4,q5
    joint_angles_A1 = np.concatenate((q_1_2_3_A1, np.array([q4_A1,q5_A1,q6_A1])), axis=None)
    joint_angles_A2 = np.concatenate((q_1_2_3_A2, np.array([ q4_A2,q5_A2,q6_A2])), axis=None)

    T_A1 = FK_dh(dh_params, joint_angles_A1, 5)
    T_A2 = FK_dh(dh_params, joint_angles_A2, 5)

    try: 
        pose_A1 = get_pose_from_T(T_A1)
        if np.allclose(pos_ee, np.array(pose_A1[0:3]), rtol=1e-04, atol=1e-05, equal_nan=False):
            # print(T_A1)
            # print('\n')
            # print(np.array(pose_A1[0:3]))
            # print('\n')
            if direction == "flat":
                q7 = 0
            else:
                q7 = q6_A1
            return [q1,q2,q3,q4_A1,q7]
        else:
            q8 = error_intentional
    except:
        pose_A2 = get_pose_from_T(T_A2)
        # print(T_A2)
        # print('\n')
        # print(np.array(pose_A2[0:3]))
        # print('\n')
        if direction == "flat":
            q7 = 0
        else:
            q7 = q6_A2
        return [q1,q2,q3,q4_A2,q7]

    # if direction=="down":
    #     q4 = q2-q3-math.pi/2  
    #     q5 = q1
    # elif direction=="flat":
    #     q4 = q2-q3
    #     q5 = 0

    # return [q1_f,q2_f,q3_f,q4_f,q5_f]

def IK_geometric_event_1(dh_params, T, direction):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array 
    @param      direction  The direction of EE, it could be "flat" or "down"

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    pose = get_pose_from_T(T)
    phi = pose[3]
    theta = pose[4]
    psi = pose[5]
    # Defining co
    l1 = dh_params[1][0] # 205.73
    l2 = dh_params[2][0] # 200
    d1 = dh_params[0][2] # 103.91
    link6_len = dh_params[4][2]; 
    angle_offset = math.pi/2 - math.atan2(50,200)
    # print(l1,l2,d1,link6_len,angle_offset)
    # print('\n')

    # End Effector Location
    pos_ee = np.array([pose[0],pose[1],pose[2]])
    x = pos_ee[0]
    y = pos_ee[1]
    z = pos_ee[2]
    # print(pos_ee)
    # print('\n')

    if direction=="down":
        pos_wrist = pos_ee + np.array([0,0,link6_len])
        # print(pos_wrist)
        # print('\n')
        # print(link6_len)
        # print('\n')
        # R_0_5 = get_R_from_euler_angles(0.0,np.pi,0.0)
        R_0_5 = get_R_from_euler_angles(phi,theta,psi)
    elif direction=="flat":
        R_0_5 = get_R_from_euler_angles(math.atan2(pose[1],pose[0]),np.pi/2,0.0)
        pos_wrist = pos_ee - link6_len*np.matmul(R_0_5,np.array([0,0,1]))

    # Wrist location
    ox = pos_wrist[0]
    oy = pos_wrist[1]
    oz = pos_wrist[2]
    # print(ox,oy,oz)
    # print('\n')
    planar_x = math.sqrt(ox**2 + oy**2)
    planar_y = oz - d1
    # print(planar_x)
    # print('\n')
    # print(planar_y)
    # print('\n')
    # Q1 - Calculation 
    q1 = math.atan2(oy,ox) - np.pi/2.0
    if q1 < -np.pi:
        q1 = q1 + 2*np.pi

    # Q3 - Calculation
    c3 = ((planar_x**2 + planar_y**2) - l1**2 -l2**2)/(2*l1*l2)
    
    if np.absolute(c3) > 1:
        print('Postion Unreachable')
    elif c3 ==1:
        theta_2 = math.atan2(oy,ox)
        theta_3 = 0
        theta_3_alt = 0
    elif c3 ==-1 and (planar_x**2 + planar_y**2) != 0:
        theta_2 = math.atan2(oy,ox)
        theta_3 = math.pi
        theta_3_alt = -math.pi
    elif c3 ==-1 and (planar_x**2 + planar_y**2) == 0:
        theta_2 = 0 # picking zero, as it has infinite solutions
        theta_3 = math.pi
        theta_3_alt = -math.pi
        print('Infinite solutions')
    else:
        theta_3 = math.acos(c3)
        theta_3_alt = -math.acos(c3)

    theta_3 = -theta_3    # Choosing Elbow Up 
    q3 = theta_3 + angle_offset
    q3_alt = theta_3_alt + angle_offset

    # Q2 - Calculation
    theta_2 = math.atan2(planar_y,planar_x) - math.atan2(l2*math.sin(theta_3),l1+l2*math.cos(theta_3))
    theta_2_alt = math.atan2(planar_y,planar_x) - math.atan2(l2*math.sin(theta_3_alt),l1+l2*math.cos(theta_3_alt))
    q2 = angle_offset - theta_2
    q2_alt = angle_offset - theta_2_alt

    q_1_2_3_A1 = np.array([q1,q2,q3])
    q_1_2_3_A2 = np.array([q1,q2_alt,q3_alt])

    # For A1 - For Q3, Q4, Q5 calculations
    joint_angles_mod_A1 = np.array([q1,q2,q3,0.0,0.0])
    T_0_3_A1 = FK_dh(dh_params, joint_angles_mod_A1, 3)
    R_0_3_A1 = T_0_3_A1[0:3,0:3]
    R_3_5_A1 = np.matmul(np.linalg.inv(R_0_3_A1), R_0_5)

    # For A2 - For Q3, Q4, Q5 calculations
    joint_angles_mod_A2 = np.array([q1,q2_alt,q3_alt,0.0,0.0])
    T_0_3_A2 = FK_dh(dh_params, joint_angles_mod_A2, 3)
    R_0_3_A2 = T_0_3_A2[0:3,0:3]
    R_3_5_A2 = np.matmul(np.linalg.inv(R_0_3_A2), R_0_5)

    q4_A1,q5_A1,q6_A1 = get_euler_angles_from_T(R_3_5_A1)
    q4_A2,q5_A2,q6_A2 = get_euler_angles_from_T(R_3_5_A2)

    # Checking q1,q2,q3,q4,q5
    joint_angles_A1 = np.concatenate((q_1_2_3_A1, np.array([q4_A1,q5_A1,q6_A1])), axis=None)
    joint_angles_A2 = np.concatenate((q_1_2_3_A2, np.array([ q4_A2,q5_A2,q6_A2])), axis=None)

    T_A1 = FK_dh(dh_params, joint_angles_A1, 5)
    T_A2 = FK_dh(dh_params, joint_angles_A2, 5)

    try: 
        pose_A1 = get_pose_from_T(T_A1)
        if np.allclose(pos_ee, np.array(pose_A1[0:3]), rtol=1e-04, atol=1e-05, equal_nan=False):
            # print(T_A1)
            # print('\n')
            # print(np.array(pose_A1[0:3]))
            # print('\n')
            if direction == "flat":
                q7 = 0
            else:
                q7 = q6_A1
            return [q1,q2,q3,q4_A1,q7]
        else:
            q8 = error_intentional
    except:
        pose_A2 = get_pose_from_T(T_A2)
        # print(T_A2)
        # print('\n')
        # print(np.array(pose_A2[0:3]))
        # print('\n')
        if direction == "flat":
            q7 = 0
        else:
            q7 = q6_A2
        return [q1,q2,q3,q4_A2,q7]

    # if direction=="down":
    #     q4 = q2-q3-math.pi/2  
    #     q5 = q1
    # elif direction=="flat":
    #     q4 = q2-q3
    #     q5 = 0

    # return [q1_f,q2_f,q3_f,q4_f,q5_f]


def rotation_matrix(theta1, theta2, theta3, order='xyz'):
    """
    input
        theta1, theta2, theta3 = rotation angles in rotation order (degrees)
        oreder = rotation order of x,y,z e.g. XZY rotation -- 'xzy'
    output
        3x3 rotation matrix (numpy array)
    """
    c1 = np.cos(theta1 * np.pi / 180)
    s1 = np.sin(theta1 * np.pi / 180)
    c2 = np.cos(theta2 * np.pi / 180)
    s2 = np.sin(theta2 * np.pi / 180)
    c3 = np.cos(theta3 * np.pi / 180)
    s3 = np.sin(theta3 * np.pi / 180)

    if order == 'xzx':
        matrix=np.array([[c2, -c3*s2, s2*s3],
                         [c1*s2, c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3],
                         [s1*s2, c1*s3+c2*c3*s1, c1*c3-c2*s1*s3]])
    elif order=='xyx':
        matrix=np.array([[c2, s2*s3, c3*s2],
                         [s1*s2, c1*c3-c2*s1*s3, -c1*s3-c2*c3*s1],
                         [-c1*s2, c3*s1+c1*c2*s3, c1*c2*c3-s1*s3]])
    elif order=='yxy':
        matrix=np.array([[c1*c3-c2*s1*s3, s1*s2, c1*s3+c2*c3*s1],
                         [s2*s3, c2, -c3*s2],
                         [-c3*s1-c1*c2*s3, c1*s2, c1*c2*c3-s1*s3]])
    elif order=='yzy':
        matrix=np.array([[c1*c2*c3-s1*s3, -c1*s2, c3*s1+c1*c2*s3],
                         [c3*s2, c2, s2*s3],
                         [-c1*s3-c2*c3*s1, s1*s2, c1*c3-c2*s1*s3]])
    elif order=='zyz':
        matrix=np.array([[c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3, c1*s2],
                         [c1*s3+c2*c3*s1, c1*c3-c2*s1*s3, s1*s2],
                         [-c3*s2, s2*s3, c2]])
    elif order=='zxz':
        matrix=np.array([[c1*c3-c2*s1*s3, -c1*s3-c2*c3*s1, s1*s2],
                         [c3*s1+c1*c2*s3, c1*c2*c3-s1*s3, -c1*s2],
                         [s2*s3, c3*s2, c2]])
    elif order=='xyz':
        matrix=np.array([[c2*c3, -c2*s3, s2],
                         [c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1],
                         [s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2]])
    elif order=='xzy':
        matrix=np.array([[c2*c3, -s2, c2*s3],
                         [s1*s3+c1*c3*s2, c1*c2, c1*s2*s3-c3*s1],
                         [c3*s1*s2-c1*s3, c2*s1, c1*c3+s1*s2*s3]])
    elif order=='yxz':
        matrix=np.array([[c1*c3+s1*s2*s3, c3*s1*s2-c1*s3, c2*s1],
                         [c2*s3, c2*c3, -s2],
                         [c1*s2*s3-c3*s1, c1*c3*s2+s1*s3, c1*c2]])
    elif order=='yzx':
        matrix=np.array([[c1*c2, s1*s3-c1*c3*s2, c3*s1+c1*s2*s3],
                         [s2, c2*c3, -c2*s3],
                         [-c2*s1, c1*s3+c3*s1*s2, c1*c3-s1*s2*s3]])
    elif order=='zyx':
        matrix=np.array([[c1*c2, c1*s2*s3-c3*s1, s1*s3+c1*c3*s2],
                         [c2*s1, c1*c3+s1*s2*s3, c3*s1*s2-c1*s3],
                         [-s2, c2*s3, c2*c3]])
    elif order=='zxy':
        matrix=np.array([[c1*c3-s1*s2*s3, -c2*s1, c1*s3+c3*s1*s2],
                         [c3*s1+c1*s2*s3, c1*c2, s1*s3-c1*c3*s2],
                         [-c2*s3, s2, c2*c3]])

    return matrix