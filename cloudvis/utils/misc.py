import math
import numpy as np
def find_nearest_in_sorted(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

def se3_inv(mtx):
    mtx = mtx.copy()
    R = mtx[:3, :3].copy()
    t = mtx[:3, 3].copy()
    mtx[:3, :3] = R.T
    mtx[:3, 3] = -(R.T).dot(t)
    return mtx