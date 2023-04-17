import sys 
import time
sys.path.append("..") 
import cv2
import math
from objects import Camera
import autograd.numpy as np
from autograd import jacobian
from run_vp_detect import VP
from init import *
# from class_ekf_chi import *
from EKF_with_6_state import *
from scipy.spatial.transform import Rotation 
from utils import *
from feature_manage import feature_management
import matplotlib.pyplot as plt

np.random.seed(0)

colours = 255 * np.eye(3)
colours = colours[:, ::-1].astype(int).tolist()

def init():
    # Read the data from file 
    IMU_data = np.genfromtxt('Bicocca_2009-02-25b-IMU_STRETCHED.csv', delimiter=',')

    img_imu_pair = []
    with open('img_imu.txt', 'r') as f:
        for line in f:
            img_idx, imu_idx = eval(line.rstrip())
            img_imu_pair.append((f"{float(img_idx):0>16.6f}", imu_idx)) 
    
    img_od_pair = []
    with open('img_odx.txt', 'r') as f:
        for line in f:
            img_idx, vx, vy, vz = eval(line.rstrip())
            img_od_pair.append((img_idx, vx, vy, vz)) 

    return IMU_data, img_imu_pair, img_od_pair

if __name__ == "__main__":
    # Camera
    cam = Camera()
    K = cam.intrinsic()
    K_inv = np.linalg.inv(K)

    # Rotation matrix & position
    # position = np.array([-0.624,-8.987,0]).reshape(-1,1)
    position = np.array([0,0,0]).reshape(-1,1)

    # img, imu, od
    IMU_data, img_imu_pair, img_od_pair = init() 

    """ 1. Initialization of the state vector """
    xc_list = []
    features = [] # maintain 15 features
    flag = 0
    for img_num in range(500, len(img_imu_pair)):
        # # img
        filename = "lu_vp_detect/FRONTAL/FRONTAL_" + img_imu_pair[img_num][0] + ".png"

        if flag == 0:
            if filename != "lu_vp_detect/FRONTAL/FRONTAL_1235604664.906455.png":
                continue
            else:
                flag = 1

        # rotation:R, quaternion:qwc
        first_col = IMU_data[:,0]
        idx = np.where(first_col == float(img_imu_pair[img_num][1]))[0].item()
        R = IMU_data[idx][-9:].reshape(3,3)

        r = Rotation.from_matrix(R)
        # q_wc = r.as_quat()

        # velocity: v_w, angular velocity: w_c
        v_w = np.array(img_od_pair[img_num][1:])[:2]
        w_c = IMU_data[idx][7]

        # xc = np.hstack((position.T.squeeze(), q_wc, v_w, w_c))
        xc = np.hstack((position.T.squeeze(), v_w, w_c))
        xc_list.append(xc)

        cov_cam = np.eye(6)

        linSeg, vp2, vp3, img = VP(filename)
        struct_lines = initialization(img, linSeg, vp2, R, K_inv, position, cov_cam) 
        vp, li, lbar = measurement(struct_lines, position, R, K)  # calculate projected 


        # for j in range(len(struct_lines)):
        #     matrix = np.eye(3)
        #     if all(matrix[0].reshape(3,1) == struct_lines[j].pi):
        #         color = colours[0]
        #     elif all(matrix[2].reshape(3,1) == struct_lines[j].pi):
        #         color = colours[1]
        #     else:
        #         color = colours[2]   

        #     b = cv2.line(img, (int(li[j][0,0]), int(li[j][1,0])), (int(vp[j][0,0]), int(vp[j][1,0])),
        #         color, 2, cv2.LINE_AA)
            # b = cv2.circle(img, (int(li[j][0,0]), int(li[j][1,0])), radius=3, color = (0,0,225), thickness=4)
        
        # cv2.imshow('rew', img)
        # cv2.waitKey(0)

        while len(features) != 15 and struct_lines:
            features.append(struct_lines.pop())
        
        if len(features) == 15:
            break
    
    # The start index for image detection
    next_idx = img_num+1

    xc = np.mean(np.array(xc_list), axis=0)
    State_vector = np.zeros(6+4*15)
    State_vector[:6] = xc
    
    Covariance_matrix = np.eye(6+4*15)
    for i in range(15):
        State_vector[6+i*4:6+i*4+4] = [features[i].ca.item(), features[i].cb.item(), features[i].theta.item(), features[i].h.item()]

        Covariance_matrix[6+i*4:6+i*4+4, 6+i*4:6+i*4+4] = features[i].cov_ll
        Covariance_matrix[6+i*4:6+i*4+4, 0:6] = features[i].cov_lx 
        Covariance_matrix[0:6, 6+i*4:6+i*4+4] = features[i].cov_lx.T

    # sigma_noise = np.random.randn(6, 6) # motion noise
               
    # plt.ion()                  
    """ 2. Detect new segment """
    ####
    Test = []
    xx = []
    yy = []
    fig, ax = plt.subplots()
    ###
    for img_num in range(next_idx, len(img_imu_pair)):
        if img_num % 60 != 0:
            continue
        
        # img
        filename = "lu_vp_detect/FRONTAL/FRONTAL_" + img_imu_pair[img_num][0] + ".png"   

        if filename == "lu_vp_detect/FRONTAL/FRONTAL_1235604705.761668.png":
            break
        # if img_num == 600:
        #     break

        # rotation:R
        first_col = IMU_data[:,0]
        idx = np.where(first_col == float(img_imu_pair[img_num][1]))[0].item()
        R = IMU_data[idx][-9:].reshape(3,3)

        # velocity: v_w, angular velocity: w_c
        linSeg_v_w = np.array(img_od_pair[img_num][1:-1])
        linSeg_w_c = IMU_data[idx][7]
        State_vector[3:5] = linSeg_v_w 
        State_vector[5] = linSeg_w_c 

        linSeg, vp2, vp3, img = VP(filename)

        """ 2.1 EKF prediction """
        ## add a zero (x1,y1,0),(x2,y2,0)  

        m, n = len(linSeg), 15 
        linSeg_zero = np.zeros((linSeg.shape[0], 6))
        linSeg_zero[:, :2] = linSeg[:,:2]
        linSeg_zero[:, 3:5] = linSeg[:,-2:]

        ## structure line 15x4
        l = np.array([[line.ca.item(), line.cb.item(), line.theta.item(), line.h.item()] for line in features])
        ## dominant direction 15x3
        eta = np.array([line.eta.squeeze() for line in features])
        ## Projection matrix 15x2x3
        P = np.array([line.P.T for line in features])
        ## pi plane of all structure lines 15x3
        pi = np.array([line.pi.squeeze() for line in features])
        ## lbar of all structure lines
        lbar = np.array([line.lbar.squeeze() for line in features])
        ## vanishing pts of all structure lines
        vp_sl = np.array([np.array(line.vp).squeeze() for line in features])
        # covariance of noise
        N = np.diag([0.00025]*6)
        big_N = np.diag([0.00025]*2*m*n)

        # State_vector = State_vector[:2] 
        ekf = EKF(l, eta, P, pi, lbar, vp_sl, State_vector.reshape(-1, 1), Covariance_matrix, N, big_N, linSeg_zero, K, R, delta_t=2, n_delta=0.01)
        # run motion model to get pred state and pred sigma
        ekf.motion_model()

        """ 2.2 data association """
        # Chi square distance
        selected_linSeg = ekf.chi_square_distance()
        selected_linSeg = np.array(selected_linSeg).reshape(-1,6)
        linSeg = np.hstack((selected_linSeg[:,0:2], selected_linSeg[:,3:5]))

        # ZNCC
        line_list = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for i in range(len(linSeg)):
            x = (linSeg[i][0] + linSeg[i][2]) / 2
            y = (linSeg[i][1] + linSeg[i][3]) / 2
            line_patch = patch_mid(gray, x, y) 
            for f in features:
                if ZNCC(f.patch, line_patch) >= 0.8:
                    f.NoF = 0
                    line_list.append(i)
                    break
        for f in features:
            if f.NoF == -1:
                f.NoF = 1
            elif f.NoF > 0:
                f.NoF += 1 

        linSeg = linSeg[line_list]

        linSeg_zero = np.zeros((linSeg.shape[0], 6))
        linSeg_zero[:, :2] = linSeg[:,:2]
        linSeg_zero[:, 3:5] = linSeg[:,-2:]  
        
        # run measurement model to get new state and new sigma
        ekf.N = np.random.randn(2*len(linSeg_zero)*15, 2*len(linSeg_zero)*15)
        pred_mu, pred_sigma = ekf.measurement_model(linSeg_zero)

        State_vector[:6] = pred_mu[:6].squeeze()
        Covariance_matrix[:6, :6] = pred_sigma[:6, :6] 
        position = np.vstack((State_vector[:2].reshape(-1,1), np.array([0])))

        print(position)   

        #### plotting ####
        xx.append(position[0,0].item())
        yy.append(position[1,0].item())     
        ax.cla() # clear plot
        ax.plot(xx, yy, 'o-') # draw line chart
        plt.pause(0.1)

        cov_cam = Covariance_matrix[:6, :6] 

        Test.append(position)
        """ Feature management """
        linSeg = np.hstack((linSeg_zero[:,0:2], linSeg_zero[:,3:5])) 
        struct_lines = initialization(img, linSeg, vp2, R, K_inv, position, cov_cam) 
        vp, li, lbar = measurement(struct_lines, position, R, K)  # calculate projected 

        feature_management(features, State_vector, Covariance_matrix, struct_lines)

with open('test.txt', 'w') as f:
    f.write(str(Test) + '\n')

# ax.append([Test[i][0].item() for i in range(len(Test))])           
# ay.append([Test[i][0].item() for i in range(len(Test))])       
# plt.clf()              
# plt.plot(ax[0],ay[0], 'o-')       
# plt.pause(5)         
# plt.ioff()  