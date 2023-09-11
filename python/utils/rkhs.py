import numpy as np
from kernel import kernel_function
import ipdb
from compute_diff_norm import compute_diff_norm
from scipy.io import savemat

def gen_rkhs_function(data, l, num_points, rkhs_index, kernel_index, num_functions):

    """
    @param data: list or array - [[]]  dim x arms - f_t received signal for each arm - dim=1
    @param l: scale paramaeter l>0
    @param num_points: 
    @param rkhs_index: select between RKHS function with different coefficients
                      use rkhs_index = 1 for the coefficients in [-1,1] and rkhs_index = 2 for the coefficients in [0,1]
    @param kernel_index: select between different kernels which are squared exponential kernel
                 and Maternel kernel (more options can be added as well) --> from kernel.py
    @param num_functions: required to generate f_test
    :return: ||f_test||_RKHS , sample points and weights
    """

    if not isinstance(data, np.ndarray):
        data_arr = np.array(data)
    else:
        data_arr = data

    dim = data_arr.shape[0]
    num_arms = data_arr.shape[1]


    K = num_functions
    # ipdb.set_trace()
    x_j_vec = np.zeros([num_points,K,dim])
    ind = 0
    for k in range(K):
        rand_d = np.random.rand(num_points, dim)    # uniform distribution in the interval (0,1)
        # print(rand_d)
        for d in range(num_points):
            x_j_vec[d][k] = rand_d[d]
            # print(x_j_vec)


    a_j_vec = np.zeros([K, num_points])
    f = np.zeros([K,num_arms])
    
    for k in range(K):
        if rkhs_index == 1:
            a_j_vec[k] = -1 + 2 * np.random.rand(1, num_points)
            # print(a_j_vec)
        else:
            a_j_vec[k] = np.random.rand(1, num_points)
            # print(a_j_vec)
        for i in range(num_arms):
            if dim == 1:
                x = data_arr[0][i]
            else:
                x = np.array([data[row][i] for row in range(dim)])
            for j in range(num_points):
                # ipdb.set_trace()
                y = x_j_vec[j][k]
                f[k][i] = f[k][i] + a_j_vec[k][j] * kernel_function(x,y,l,kernel_index)

    return f, x_j_vec, a_j_vec

if __name__=="__main__":
    np.random.seed(1337)

    # data1 = [[1,2,3]]
    # f, x, a = gen_rkhs_function(data1,1,3,1,1,4)
    # print(f)
    # print(x)
    # print(a)

    d = 1
    n_arms = 26
    num_func = 2

    data = np.zeros((d,n_arms))
    for i in range(d):
        for j in range(n_arms):
            data[i][j] = (1 / n_arms) * (j+1)

    #generate kernel
    kernel_index = 1
    l = 0.2
    kernel_x = np.zeros((n_arms,n_arms))
    kernel_y = np.zeros((n_arms,n_arms))
    for i in range(n_arms):
        for j in range(n_arms):
            x = data[0][i]
            y = data[0][j]
            kernel_x[i][j] = kernel_function([x],[y],l,kernel_index)
            kernel_y[i][j] = kernel_function([x],[y],l,kernel_index)
    
    # ipdb.set_trace()
    rkhs_index = 1
    f_test, sample_points, weights = gen_rkhs_function(data, l, n_arms, rkhs_index, kernel_index, num_func)
    weight_x_1 = weights[0]
    weight_x_2 = weights[1]
    diff_x_1 = compute_diff_norm(weight_x_1, weight_x_2, sample_points, l, kernel_index)

    g_test, sample_points, weights = gen_rkhs_function(data, l, n_arms, rkhs_index, kernel_index, num_func)
    weight_y_1 = weights[0]
    weight_y_2 = weights[1]
    diff_y_1 = compute_diff_norm(weight_y_1, weight_y_2, sample_points, l, kernel_index)

    savemat("data/diff_norm_fg.mat", {'diff_fg': [diff_x_1, diff_y_1]})