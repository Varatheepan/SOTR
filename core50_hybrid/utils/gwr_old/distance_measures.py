############################################################
# Distance measures needed for self-organization networks
# Duvindu Piyasena : 11/9/2019
############################################################
import numpy as np

#-----------------------------------------
# Manhattan distance based on numpy
# 
# Manhattan dist = sum(abs(a-b))
#-----------------------------------------
def manhattan_distance(a, b, enable_quantization=False) : 
    if(enable_quantization == True) :  
        a = a.astype('int16')
        b = b.astype('int16')
        invalid_idx_a = np.where(a>255)
        invalid_idx_b = np.where(b>255)
        if(invalid_idx_a[0].size > 0):
            print("invalid a : ", a[invalid_idx_a])
            raw_input()
        if(invalid_idx_b[0].size > 0):
            print("invalid b : ", b[invalid_idx_b])
            raw_input()
    c = np.abs(a-b)
    d = np.sum(c)
    return d

def manhattan_distance_test(a, b, enable_quantization=False) : 
    if(enable_quantization == True) :  
        a = a.astype('int16')
        b = b.astype('int16')
        invalid_idx_a = np.where(a>255)
        invalid_idx_b = np.where(b>255)
        if(invalid_idx_a[0].size > 0):
            print("invalid a : ", a[invalid_idx_a])
            raw_input()
        if(invalid_idx_b[0].size > 0):
            print("invalid b : ", b[invalid_idx_b])
            raw_input()
    c = np.abs(a-b)
    d = np.sum(c)
    return d
