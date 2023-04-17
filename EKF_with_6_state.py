import autograd.numpy as np # Thinly−wrapped numpy
from autograd import grad, jacobian # grad(f) returns f’
from scipy.linalg import block_diag
from utils_1 import *
np.random.seed(0)

class EKF:

    def __init__(self, l, eta, P, pi, lbar, vp, mu, sigma, N, big_N, linSeg_zero, camIntrinsic_K, R_wc, delta_t=None, n_delta=None):
        self.l = l  # ca, cb, theta, h  (shape 15 x 4)
        self.eta = eta  # dominant direction of structure lines (shape 15 x 3)
        self.P = P  # 3d to 2d matrix (shape 15 x 2 x 3)
        self.pi = pi  # pi plane of structure lines (shape 15 x 3)
        self.lbar = lbar  # lbar of structure lines (shape 15 x 3)
        self.vp = vp  # vanishing pts (shape 15 x 3)
        self.mu = mu  # state (shape 73 x 1) +++++++ NEW SHAPE(66 x1) 
        self.sigma = sigma  # covariance (shape 73 x 73) ++++++ NEW SHAPE (66 x 1)
        self.delta_t = delta_t  # delta t in dynamic model
        self.n_delta = n_delta  # delta in noise matrix
        self.sigma_noise = N  # covariance of noise (shape 6 x 6) ++++++++ NEW shape (3 x 3)
        self.line_segment_set = linSeg_zero  # set of line segments (shape m x 6)
        self.N = None
        self.camIntrinsic_K = camIntrinsic_K
        self.R_wc = R_wc
        self.new_line_seg_set = []

    #### motion model ####
    
    def fc_Xc(self, p_w, v_w, theta, w_c):
        # dynamic model
        pred_p_w = p_w + v_w * self.delta_t # prediction of position
        pred_theta = theta + w_c * self.delta_t
        pred_v_w = v_w # prediction of velocity
        pred_w_c = w_c # prediction of angular velocity
        return np.vstack((pred_p_w, pred_theta, pred_v_w, pred_w_c)) # fc_Xc
    
    def Jacobian_Fx(self, n):
        Jac_fc_xc = [[1, 0, 0, self.delta_t, 0, 0],
                     [0, 1, 0, 0, self.delta_t, 0],
                     [0, 0, 1, 0, 0, self.delta_t],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]]
        return block_diag(np.array(Jac_fc_xc), np.eye(4 * n))
    
    def Jacobian_Fn(self, p_w, v_w, q_wc, w_c, n):
        Jac_fc_n = [[self.delta_t, 0, 0, 0, 0, 0],
                    [0, self.delta_t, 0, 0, 0, 0],
                    [0, 0, 0, self.delta_t, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1 / 2 * q_wc[1, 0] * self.delta_t, 0, 0],
                    [0, 0, 0, 0, 1 / 2 * q_wc[2, 0] * self.delta_t, 0],
                    [0, 0, 0, 0, 0, 1 / 2 * q_wc[3, 0] * self.delta_t],
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1]]
        return np.vstack((1 / self.n_delta * np.array(Jac_fc_n), np.zeros((4 * n, 6))))
    
    def motion_model(self):
        n = (len(self.sigma) - 6) // 4
        p_w, theta, v_w, w_c = self.mu[:2], self.mu[2], self.mu[3:5], self.mu[5]
        pred_Xc = self.fc_Xc(p_w, v_w, theta, w_c) # prediction of state
        self.pred_mu = np.vstack((pred_Xc, self.mu[6:]))
        Fx = self.Jacobian_Fx(n) # Eq 8
        # Fn = self.Jacobian_Fn(p_w, v_w, q_wc, w_c, n) # Eq 8
        self.pred_Sigma = Fx @ self.sigma @ Fx.T #prediction of covariance
        return self.pred_mu, self.pred_Sigma
    
    #### measurement model ####
    
    def elmt_m_ij_pw(self, p_w):
        '''
        Return: m_ij, shape in 1 x 2, only for computing Jacobian
        '''

        # midpoint of the line segment
        mid_P = (self.line_seg[:3] + self.line_seg[3:]) / 2 # compute from self.line_seg
        l = np.random.rand(1, 4) # need to be stored in self.structure_line
        
        # dominant direction
        dominant_D = self.dom_D
        pi_plane = self.pi_plane
        # vanishing point
        v = self.v
        # tranform 3d to 2d matrix
        P = self.P_mtx

        ## Global varibles
        # intrinsic martrix
        camIntrinsic_K = self.camIntrinsic_K
        # rotation matrix
        R_wc = self.R_wc
        R_cw = R_wc.T

        pw0, pw1 = p_w
        pw2 = 0

        # Eq 17
        temp =  R_wc @ np.linalg.inv(camIntrinsic_K) @ mid_P
        m0 = temp[0] + pw0
        m1 = temp[1] + pw1
        m2 = temp[2] + pw2

        # Eq 18
        lw0 = (m0 * dominant_D[1] - m1 * dominant_D[0]) * pi_plane[1] + (m0 * dominant_D[2] - m2 * dominant_D[0]) * pi_plane[2]
        lw1 = (m1 * dominant_D[0] - m0 * dominant_D[1]) * pi_plane[0] + (m1 * dominant_D[2] - m2 * dominant_D[1]) * pi_plane[2]
        lw2 = (m2 * dominant_D[0] - m0 * dominant_D[2]) * pi_plane[0] + (m2 * dominant_D[1] - m1 * dominant_D[2]) * pi_plane[1]

        # Eq 19
        lp0 = P[0, 0] * lw0 + P[0, 1] * lw1 + P[0, 2] * lw2
        lp1 = P[1, 0] * lw0 + P[1, 1] * lw1 + P[1, 2] * lw2

        # Eq 20
        c0 = pw0
        c1 = pw1
        c2 = pw2
        ow0 = (c0 * dominant_D[1] - c1 * dominant_D[0]) * pi_plane[1] + (c0 * dominant_D[2] - c2 * dominant_D[0]) * pi_plane[2]
        ow1 = (c1 * dominant_D[0] - c0 * dominant_D[1]) * pi_plane[0] + (c1 * dominant_D[2] - c2 * dominant_D[1]) * pi_plane[2]
        ow2 = (c2 * dominant_D[0] - c0 * dominant_D[0]) * pi_plane[0] + (c2 * dominant_D[1] - c1 * dominant_D[2]) * pi_plane[1]

        # Eq 21
        op0 = P[0, 0] * ow0 + P[0, 1] * ow1 + P[0, 2] * ow2
        op1 = P[1, 0] * ow0 + P[1, 1] * ow1 + P[1, 2] * ow2

        # Eq 22
        c_a = op0
        c_b = op1
        theta = np.arctan2(lp1 - op1, lp0 - op0)
        h = 1 / np.sqrt((lw0 - ow0) ** 2 + (lw1 - ow1) ** 2 + (lw2 - ow2) ** 2)

        # Eq 27
        l_wh0 = P[0, 0] * (c_a * h + np.cos(theta)) + P[1, 0] * (c_b * h + np.sin(theta))
        l_wh1 = P[0, 1] * (c_a * h + np.cos(theta)) + P[1, 1] * (c_b * h + np.sin(theta))
        l_wh2 = P[0, 2] * (c_a * h + np.cos(theta)) + P[1, 2] * (c_b * h + np.sin(theta))

        # Eq 28
        l_c0 = R_cw[0, 0] * (l_wh0 - pw0 * h) + R_cw[0, 1] * (l_wh1 - pw1 * h) + R_cw[0, 2] * (l_wh2 - pw2 * h)
        l_c1 = R_cw[1, 0] * (l_wh0 - pw0 * h) + R_cw[1, 1] * (l_wh1 - pw1 * h) + R_cw[1, 2] * (l_wh2 - pw2 * h)
        l_c2 = R_cw[2, 0] * (l_wh0 - pw0 * h) + R_cw[2, 1] * (l_wh1 - pw1 * h) + R_cw[2, 2] * (l_wh2 - pw2 * h)

        # Eq 29
        l_i0 = camIntrinsic_K[0, 0] * l_c0 + camIntrinsic_K[0, 1] * l_c1 + camIntrinsic_K[0, 2] * l_c2
        l_i1 = camIntrinsic_K[1, 0] * l_c0 + camIntrinsic_K[1, 1] * l_c1 + camIntrinsic_K[1, 2] * l_c2
        l_i2 = camIntrinsic_K[2, 0] * l_c0 + camIntrinsic_K[2, 1] * l_c1 + camIntrinsic_K[2, 2] * l_c2

        # Eq 30
        l_hat0 = v[1] * l_i2 - v[2] * l_i1
        l_hat1 = v[2] * l_i0 - v[0] * l_i2
        l_hat2 = v[0] * l_i1 - v[1] * l_i0

        s_a, s_b = self.line_seg[:3], self.line_seg[3:]
        m_ij = np.array([(s_a[0] * l_hat0 + s_a[1] * l_hat1 + s_a[2] * l_hat2) / np.sqrt(l_hat0 ** 2 + l_hat1 ** 2),\
                         (s_b[0] * l_hat0 + s_b[1] * l_hat1 + s_b[2] * l_hat2) / np.sqrt(l_hat0 ** 2 + l_hat1 ** 2)])
        return  m_ij

    def elmt_m_ij_l(self, l):
        '''
        Return shape 2 x 1, only for computing Jacobian
        '''

        # vanishing point
        v = self.v
        # tranform 3d to 2d matrix
        P = self.P_mtx

        ## Gobal varibles
        # intrinsic martrix
        camIntrinsic_K = self.camIntrinsic_K
        # rotation matrix
        R_wc = self.R_wc
        # R_cw = R_wc.T
        R_cw = R_wc.T
        pw0, pw1 = self.pred_mu[:2, 0]
        pw2 = 0

        ca, cb, theta, h = l

        # Eq 27
        l_wh0 = P[0, 0] * (ca * h + np.cos(theta)) + P[1, 0] * (cb * h + np.sin(theta))
        l_wh1 = P[0, 1] * (ca * h + np.cos(theta)) + P[1, 1] * (cb * h + np.sin(theta))
        l_wh2 = P[0, 2] * (ca * h + np.cos(theta)) + P[1, 2] * (cb * h + np.sin(theta))

        # Eq 28
        l_c0 = R_cw[0, 0] * (l_wh0 - pw0 * h) + R_cw[0, 1] * (l_wh1 - pw1 * h) + R_cw[0, 2] * (l_wh2 - pw2 * h)
        l_c1 = R_cw[1, 0] * (l_wh0 - pw0 * h) + R_cw[1, 1] * (l_wh1 - pw1 * h) + R_cw[1, 2] * (l_wh2 - pw2 * h)
        l_c2 = R_cw[2, 0] * (l_wh0 - pw0 * h) + R_cw[2, 1] * (l_wh1 - pw1 * h) + R_cw[2, 2] * (l_wh2 - pw2 * h)

        # Eq 29
        l_i0 = camIntrinsic_K[0, 0] * l_c0 + camIntrinsic_K[0, 1] * l_c1 + camIntrinsic_K[0, 2] * l_c2
        l_i1 = camIntrinsic_K[1, 0] * l_c0 + camIntrinsic_K[1, 1] * l_c1 + camIntrinsic_K[1, 2] * l_c2
        l_i2 = camIntrinsic_K[2, 0] * l_c0 + camIntrinsic_K[2, 1] * l_c1 + camIntrinsic_K[2, 2] * l_c2

        # Eq 30
        l_hat0 = v[1] * l_i2 - v[2] * l_i1
        l_hat1 = v[2] * l_i0 - v[0] * l_i2
        l_hat2 = v[0] * l_i1 - v[1] * l_i0

        s_a, s_b = self.line_seg[:3], self.line_seg[3:]
        m_ij = np.array([(s_a[0] * l_hat0 + s_a[1] * l_hat1 + s_a[2] * l_hat2) / np.sqrt(l_hat0 ** 2 + l_hat1 ** 2), \
                         (s_b[0] * l_hat0 + s_b[1] * l_hat1 + s_b[2] * l_hat2) / np.sqrt(l_hat0 ** 2 + l_hat1 ** 2)])
        return  m_ij
    
    def Jacobian_pw(self, p_w):
        '''
        Return Jacobian of one line segment and one structure line
        Shape is 2 x (6 + 4 * n)
        '''
        f_pw = jacobian(self.elmt_m_ij_pw)
        return f_pw(p_w)

    def Jacobian_l(self, l):
        f_l = jacobian(self.elmt_m_ij_l)
        return f_l(l)

    def Jacobian_H(self):
        n = 15
        m = len(self.line_segment_set)
        H = np.zeros((2 * m * n, 6 + 4 * n))
        for i in range(n):
            struct_line = self.l[i, :] # shape (4, )
            self.v = self.vp[i, :] # shape (3, )
            self.dom_D = self.eta[i, :] # shape (3, )
            self.pi_plane = self.pi[i, :] # shape (3, )
            self.P_mtx = self.P[i, :, :] # shape (2, 3)

            for j in range(m):
                self.line_seg = self.line_segment_set[j]
                H_t = np.zeros((2, 6 + 4 * n))
                H_t[:, 0:2] = self.Jacobian_pw(self.pred_mu[:2, 0])
                H_t[:, 6+4*i:10+4*i] = self.Jacobian_l(struct_line)
                H[i+2*j:i+2*j+2, :] = H_t
        return H

    def r_matrix(self):
        '''
        Return residual matrix 2 * m * n by 1
        '''
        n = 15
        m = len(self.line_segment_set)
        h = []
        for i in range(n):
            lbar = self.lbar[i]
            for j in range(m):
                s_a, s_b = self.line_segment_set[j][:3], self.line_segment_set[j][3:]
                m_ij = m_x_func(lbar, s_a, s_b).reshape(2,)
                if i == 0 and j == 0:
                    h = m_ij
                else:
                    h = np.hstack((h, m_ij))
        return - np.array(h)
    
    def chi_square_distance(self):
        '''
        first step to select inliers
        '''
        n = 15
        n_sl = len(self.l)
        for line_seg in self.line_segment_set:
            l_chi_square = np.zeros(n_sl)
            self.line_seg = line_seg
            for i in range(n_sl):
                    # extract one line structure
                    # stack part of Jacobian H
                    self.v = self.vp[i, :] # shape (3, )
                    self.dom_D = self.eta[i, :] # shape (3, )
                    self.pi_plane = self.pi[i, :] # shape (3, )
                    self.P_mtx = self.P[i, :, :] # shape (2, 3)
                    r_i = m_x_func(self.lbar[i, :], line_seg[:3], line_seg[3:])
                    H_i = np.zeros((2, 6+4*n))
                    H_i[:, 0:2] = self.Jacobian_pw(self.pred_mu[:2, 0])
                    H_i[:, 6+4*i:10+4*i] = self.Jacobian_l(self.l[i, :])
                    # compute chi-square of the the selected line seg and one structure line
                    l_chi_square[i] = chi_square(r_i, H_i)
            # check whether the selected line seg could be inlier
            if np.any(l_chi_square < 5):
                # add inliers to the new set for line segments
                if len(self.new_line_seg_set) > 0:
                    self.new_line_seg_set = np.vstack((self.new_line_seg_set, line_seg))
                else:
                    self.new_line_seg_set = line_seg
        return self.new_line_seg_set 
    '''
    def RANSAC(self, res_outliers):
        
        #shape of sigma is (13 + 4 * n) by (13 + 4 * n)
        #shape of mu is (13 + 4 * n) by 1
        #Return: set of line segments after running RANSAC
        
        n = 15
        for line_seg in res_outliers:

            # add selected line_seg into line segments set
            new_line_seg_set = np.vstack(line_seg_set, line_seg)

            for i in range(n):
                # compute H according to the new line segments set
                # extract one line seg and one line structure
                # H Jacobian shape 2 by (13 + 4 * n)
                H_t = np.zeros((2, 13+4*n))
                H_t[:, 0:3] = self.Jacobian_pw(self.pred_mu[:3, 0])
                H_t[:, 13+4*i:17+4*i] = self.Jacobian_l(self.l[i, :])
            
                # residual
                r = m_x_func(self.l[i, :], line_seg[:3], line_seg[3:])
            
                # update state
                S = H_t @ sigma @ H_t.T + self.N[:2, :2] # shape of N and S is 2 by 2
                K = self.pred_Sigma @ H_t.T @ np.linalg.inv(S)
                new_mu = self.mu + K @ r
                p_w = new_mu[:3, 0]

            # predict all structure lines
            #new_l_set = initialized_structure_line(new_mu, self.l)
            new_l_set = np.random.rand(15, 4)
    
            # check chi_square (new line segment and all structure lines)
            n_ls = len(new_l_set)
            l_chi_square = np.zeros(n_ls)

            for j in range(n_ls):
                # extract one line structure
                # self.structure_line = new_l_set[i]
                # self.line_seg = line_seg
                # stack part of Jacobian H
                r_i = m_x_func(new_l_set[j], line_seg[:3], line_seg[3:])
                H_i = np.zeros((2, 13+4*n))
                H_i[:, 0:3] = self.Jacobian_pw(p_w)
                H_i[:, 13+4*j:17+4*j] = self.Jacobian_l(new_l_set[j])
                # compute chi-square of the the selected line seg and one structure line
                l_chi_square[j] = chi_square(r_i, H_i)
            # check whether the selected line seg could be inlier
            if np.any(l_chi_square < 3):
                line_seg_set = new_line_seg_set
        return line_seg_set
    '''
    def measurement_model(self, selected_line_segs):
        self.line_segment_set = selected_line_segs
        H = self.Jacobian_H()
        # innovation covariance
        s =  H @ self.pred_Sigma @ H.T
        S = s + 9 * np.eye(s.shape[0])
        # Kalman gain
        K = self.pred_Sigma @ H.T @ np.linalg.inv(S)
        # state update
        r = self.r_matrix()
        r = r.reshape(r.shape[0], 1)
        mu = self.pred_mu + K @ r
        sigma = self.pred_Sigma - K @ S @ K.T
        return mu, sigma