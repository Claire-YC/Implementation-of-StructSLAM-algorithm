import autograd.numpy as np

class Camera:
    def __init__(self):
        self._focalLength = (194.8847, 194.8847) # unit:mm
        # self._focalLength = (1732.3, 1732.3) # unit:pixel
        self._principalPoint = (172.1608, 125.0859)
        self._intrinsicMatrix = self.intrinsic()
        self._skewCoefficient = (-0.3375, 0.1391, 0.0004, 0, -0.0289)

        self._ccd = 36 # checked online
        self._imgSize = (320,240)
        self.__flPixel = self._imgSize[0] * self._focalLength[0] / self._ccd # unit pixel
    
    def intrinsic(self):
        intrinsic_matrix = np.array( \
            [[self._focalLength[0], 0, self._principalPoint[0]], \
             [0, self._focalLength[1], self._principalPoint[1]], \
             [0, 0, 1]])
        return intrinsic_matrix
    
    def prjM_pi(self) -> dict:
        """
        prjM_pi 2x3 projection matrix transforming the 3D vector to its 2D version
        """
        prjM_dict = {}
        # xy-plane: pi(0,0,1)
        prjM_dict["001"] = np.array([[1,0,0], [0,1,0]])
        # yz-plane: pi(1,0,0)
        prjM_dict["100"] = np.array([[0,1,0], [0,0,1]])
        # zx-plane: pi(0,1,0)
        prjM_dict["010"] = np.array([[1,0,0], [0,0,1]])
        return prjM_dict


class StructLine:
    def __init__(self, ca, cb, theta, h,eta, P, cov_ll, cov_lx, pi, patch, lbar=None, vp=None):
        self.ca = ca
        self.cb = cb
        self.theta = theta
        self.h = h
        self.eta = eta
        self.P = P
        self.center = np.array([self.ca,self.cb]).reshape(-1,1)
        self.cov_ll = cov_ll
        self.cov_lx = cov_lx 
        self.NoF = -1
        self.pi = pi
        self.patch = patch
        self.lbar = lbar
        self.vp = vp