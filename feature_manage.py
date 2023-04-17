def feature_management(f_structLine, State_vector, Covariance_matrix, new_structLine):
    """
    f_structLine: list of structure line features. length = 15
    Covariance_Matrix: 13+15*4 by 13+15*4 
    """
    idx = []
    for i in range(len(f_structLine)):
        if f_structLine[i].NoF == 30:
            idx.append(i)
    
    while idx and new_structLine:
        i = idx.pop()

        f_structLine[i] = new_structLine.pop() 
        State_vector[13+i*4:13+i*4+4] = [f_structLine[i].ca.item(), f_structLine[i].cb.item(), f_structLine[i].theta.item(), f_structLine[i].h.item()]

        Covariance_matrix[13+i*4:13+i*4+4, 13+i*4:13+i*4+4] = f_structLine[i].cov_ll
        Covariance_matrix[13+i*4:13+i*4+4, 0:13] = f_structLine[i].cov_lx 
        Covariance_matrix[0:13, 13+i*4:13+i*4+4] = f_structLine[i].cov_lx.T 
    
    return f_structLine, State_vector, Covariance_matrix 

