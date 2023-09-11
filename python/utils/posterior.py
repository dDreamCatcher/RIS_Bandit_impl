import numpy as np
import ipdb

def posterior_update(mu_old, cov_old, y_t, x_t, Lambda):

    """
    @param mu_old: current posterior mean as in size(n_arms,1)
    @param cov_old: current posterior covariance as in size(n_arms,n_arms) --> kernel
    @param y_t: current reward for selected arm 
    @param x_t: selected arm
    @param lambda: hyperparameter lambda
    :return: updated posterior mean and posterior covariance
    """
    # ipdb.set_trace()
    n_arms = mu_old.shape[0]
    a = cov_old[x_t] 
    b = cov_old.T[x_t]
    c = Lambda + cov_old[x_t][x_t]
    d = y_t - mu_old[x_t]

    A = np.tile(a.reshape(-1,1),(1,n_arms))
    B = np.tile(b, (n_arms,1))

    mu_new = mu_old + (d/c)*a
    cov_new = cov_old - (1/c) * (A * B)

    return mu_new, cov_new

if __name__=="__main__":
    np.random.seed(1337)

    n_arms = 10
    mu_old = np.random.random(n_arms)
    cov_old = np.random.random((n_arms, n_arms))
    y_t = 0.3902
    x_t = 6
    Lambda = 1

    print(mu_old)
    print(cov_old)

    print(posterior_update(mu_old, cov_old, y_t, x_t, Lambda))


