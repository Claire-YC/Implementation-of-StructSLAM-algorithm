from utils import *
from objects import *
from autograd import jacobian
import autograd.numpy as np
np.random.seed(0)

def initialization(img, linSeg, vp2, R, K_inv, position, cov_cam):
    eta, eta_norm = dominant_direction_of_vp(vp2, R, K_inv)

    # mid points
    midpoints = np.concatenate( (((linSeg[:,0] + linSeg[:,2])/2).reshape(-1,1), \
                                 ((linSeg[:,1] + linSeg[:,3])/2).reshape(-1,1)), axis=1)
    
    homogenized_midpoints = np.concatenate( (midpoints.T, np.ones((1, midpoints.shape[0])) ), axis=0) 
    midpoints_world_frame = R @ K_inv @ homogenized_midpoints +  position #3xn 

    struct_lines = []
    for i in range(linSeg.shape[0]):
        line = linSeg[i,:]
        line = np.array([line[2]-line[0], line[3]-line[1],1]).reshape(-1,1)
        dom_dir = getDominantDirection(line, eta, eta_norm)
        if (dom_dir is not None):
            L = midpoints_world_frame[:,i].reshape(3,1) @ dom_dir.T - dom_dir @ midpoints_world_frame[:,i].reshape(3,1).T
            pi = getParameterPlane(dom_dir)
            lw = L @ pi
            P = get2DProjectionMatrix(pi)      
            lp = P.T @ lw
            
            ow = (position @ dom_dir.T - dom_dir @ position.T)@ pi 
            op = P.T @ ow
            theta = np.arctan2(lp[1]-op[1], lp[0] - op[0])
            h = 1/np.linalg.norm(lw-ow)
            
            # for covariance
            def center(position):
                ow = (position @ dom_dir.T - dom_dir @ position.T)@ pi
                op = P.T @ ow
                return op
            
            ## dl/dx_c
            jaco_l = np.zeros((4,6))
            j_cam = jacobian(center)
            j_ab = j_cam(position)
            jaco_l[0][0:3] = j_ab[0].squeeze()
            jaco_l[1][0:3] = j_ab[1].squeeze()

            ## dl/ds
            cov_noise = np.diag([4,4,4,4])
            
            ## dl / dh
            cov_h = np.zeros((4,4))
            cov_h[-1:-1] = 1

            cov_ll = jaco_l @ cov_cam @ jaco_l.T + cov_noise + cov_h
            cov_lx = jaco_l @ cov_cam 


            # calculate the 11x11 patch between the mid point of the structure line
            x = (linSeg[i,:][0] + linSeg[i,:][2]) / 2
            y = (linSeg[i,:][1] + linSeg[i,:][3]) / 2
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            patch = patch_mid(gray, x, y)

            struct_lines.append(StructLine(op[0],op[1],theta,h,dom_dir,P, cov_ll, cov_lx, pi, patch))

    return struct_lines