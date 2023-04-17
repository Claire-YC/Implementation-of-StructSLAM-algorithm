import autograd.numpy as np # Thinly−wrapped numpy
np.random.seed(0)

def m_x_func(l, s_a, s_b):
    '''
    Return: m_ij, shape in 1 x 2
    '''
    m_ij = np.array([s_a.dot(l) / np.sqrt(l[0] ** 2 + l[1] ** 2), s_b.dot(l) / np.sqrt(l[0] ** 2 + l[1] ** 2)])
    return m_ij.reshape((2, 1))

def chi_square(r, H):
        '''
        r is a 2 by 1 vector of the residual vector in eq 14
        H is a 2 by 13 + 4 * n matrix of the Jacobian of h(x) in eq 11
        Return : value of Chi-Square between one line segement and structure line
        '''
        chi = r.T @ np.linalg.pinv(H @ H.T) @ r
        return chi
