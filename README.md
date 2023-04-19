# Implementation-of-StructSLAM-algorithm
## Creators: University of Michigan, Ann Arbor EECS 568/NAVARCH 568/ROB 530 Mobile Robotics: Group 15
Yang Cao, Sirui Wang, Madhav Rawal, Aashish Kumar, Pratik Shiveshwar

## Goal: The implementation of the paper "StructSLAM: Visual SLAM With Building Structure Lines"

## Code reference: 
We borrowed the code for vanishing point detection. Great thank you to the author Xiaohu Lu. [Vanishing Point](https://github.com/rayryeng/XiaohuLuVPDetection)

## How to use the code:
1. Download and install the code for vanishing point detection.
`pip install lu-vp-detect`
2. Download the [dataset](https://www.dropbox.com/sh/ewqhb32zqpat8rt/AACgVf1YehSYjovLkon6nK-oa/Datasets/Indoor/Bicocca_Static_Lamps/Bicocca_2009-02-25b?dl=0) <br>
Frontal, Groundtruth, IMU_Stretched, Odometry_XYZ.
3. Change the file path for dataset in main.py.

## File instruction:
1. img_imu.txt: As the data acquisition frequency of the camera and the IMU is different, we pairwise the camera frame with the closest IMU data.
2. img_odx.txt: For each camera frame, we find out the 5 closest odometry data to calculate the x and y linear velocity. 
3. EKF_with_6_state.py: Define the EKF class with motion model, measurement model, and Jacobian matrices. The camera state is a $6 \times 1$ vector, i.e. $x_c = [x, y, \theta, v_x, v_y, w_z]$.
4. feature_manage.py: To maintain the total number of features, 15 in our case, delete the old one and add newly initialized structure line updating the covariance at the same time.
5. init.py: To initialize the structure line with asscosiate covariance matrix.
6. objects.py: Define two objects camera and structure line.
7. utils.py and utils_1.py: All the utility funcions like computing the dominant direction, parameter plane, and so on.
