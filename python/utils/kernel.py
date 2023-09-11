from numpy.linalg import norm
import numpy as np
import math

def kernel_function(x,y,l,index):
    """
    @param x: array list
    @param y: array list
    @param l: scale parameter  l > 0
    @param index: select between different kernels which are squared exponential kernel
                 and Maternel kernel (more options can be added as well)
                - use index = 1 for SE kernel and index = 2 for Matern kernel
    :return: generated kernel 
    """
    # check if x and y are array
    if not isinstance(x, np.ndarray):
        x_arr = np.array(x)
    else:
        x_arr = x

    if not isinstance(y, np.ndarray):
        y_arr = np.array(y)
    else:
        y_arr = y        
    
    # generate kernel
    r = norm((x_arr-y_arr),2)   # distance between two points

    if index == 1:
        k = math.exp((-1 * math.pow(r,2))/(2 * math.pow(l,2)))          # squared exponential kernel
    else:
        k =  (1 + ((math.sqrt(5) * r)/l) + (5 * (math.pow(r,2))/(3 * math.pow(l,2)))) * math.exp(-1 * (math.sqrt(5) * r)/l)     # matern kernel (nu=2.5)

    return k


if __name__=="__main__":
    x=[1,2]
    y=[2,3]

    print(kernel_function(x,y,1,1))
    print(kernel_function(x,y,1,2))