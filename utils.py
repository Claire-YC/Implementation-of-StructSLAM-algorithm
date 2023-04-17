import autograd.numpy as np
import cv2
np.random.seed(0)

# Projection matrix
projection_matrix = {}
projection_matrix[0] = np.eye(3)[:,1:].reshape(-1,2)
projection_matrix[1] = np.array([[1,0],[0,0],[0,1]])
projection_matrix[2] = np.eye(3)[:,0:2].reshape(-1,2)
def get2DProjectionMatrix(pi):
    return projection_matrix[int(np.where(pi==1)[0])]

def dominant_direction_of_vp(vp2, R, K_inv):
    eta = []
    eta_norm = [] 
    for i in range(vp2.shape[0]):
        v = vp2[i,:].copy()
        v = np.concatenate((v,np.array([1])),axis=0).reshape(-1,1)
        dom_dir = R @ K_inv @ v

        eta.append(dom_dir)
        eta_norm.append(dom_dir/np.linalg.norm(dom_dir))
    eta = np.asarray(eta).squeeze().T
    eta_norm = np.asarray(eta_norm).squeeze().T
    return eta, eta_norm


def getDominantDirection(line, eta, eta_norm):
    line = line.copy()
    line_norm = line/np.linalg.norm(line)
    angles = np.arccos(np.clip(np.dot(line_norm.T,eta_norm), -1.0,1.0))* 180 / np.pi
    min_idx = np.argmin(angles,axis = 1)
    
    if angles[:,min_idx]>30:
        return None
    else:
        return eta[:,min_idx] 

def getParameterPlane(dom_dir):
    '''
    Assuming dominant directions are 3xn and are normalized
    return pi: 3xn
    '''
    pi = np.zeros((3,dom_dir.shape[1]))
    normals = np.eye(3)
    for i in range(dom_dir.shape[1]):
        min_idx = np.argmin(np.arccos(np.clip(np.dot(dom_dir[:,i].T,normals), -1.0,1.0)))
        pi[:,i] = normals[:,min_idx].copy()
    return pi


def measurement(struct_lines, position, R, K):
    li_list = []
    lbar_list = []
    vp_list = []
    for line in struct_lines:
        lwh = line.P @ (line.center * line.h + np.array([np.cos(line.theta),np.sin(line.theta)]).reshape(-1,1))
        lc = R.T @ lwh - R.T @ (position * line.h)
        li = K @ lc
        v = K @ R.T @ line.eta
        v[2,0] = 1
        lbar = np.cross(v.T, li.T).T

        vp_list.append(v)
        li_list.append(li)
        lbar_list.append(lbar)

        # add to the structure line 
        line.lbar = lbar
        line.vp = v
    return vp_list, li_list, lbar_list

def patch_mid(img, x, y):
    """
    Input:
        img: 320x240 the original image
        x, y: cooridinate of the mid point
    Return:
        img_patch: 11x11 image patch around the mid point of the line
    """
    h, w = img.shape[:2]
    # top-left corner 
    x1 = int(max(0, x - 5.5))
    y1 = int(max(0, y - 5.5)) 
    # bottom-right corner 
    x2 = int(min(w, x + 5.5))
    y2 = int(min(h, y + 5.5))
    # Extract the patch from the input image
    patch = img[y1:y2, x1:x2]
    # Resize the patch to 11x11
    patch = cv2.resize(patch, (11, 11))
    return patch

def ZNCC(patch1, patch2) -> float:
    """
    Input:
        img1, img2: 11x11 image patch
    Return:
        zncc_score: measures the similarity of two images[-1,1], threshold 0.8
    """
    m,n = patch1.shape
    mean_1, std_1 = np.mean(patch1), np.std(patch1)
    mean_2, std_2 = np.mean(patch2), np.std(patch2)

    # zncc_score = (correlate(img1, img2, mode='same') - mean_1*mean_2) / (std_1*std_2)
    zncc_score = 0
    for i in range(m):
        for j in range(n):
            zncc_score += (patch1[i][j] - mean_1) * (patch2[i][j] - mean_2)
    return float(zncc_score) / (m*n * std_1 * std_2)