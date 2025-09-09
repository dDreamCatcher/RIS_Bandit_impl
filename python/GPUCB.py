'''
implement Gaussian Process Upper Confidence Bound Bandit policy
ref: @misc{deng2022interference,
      title={Interference Constrained Beam Alignment for Time-Varying Channels via Kernelized Bandits}, 
      author={Yuntian Deng and Xingyu Zhou and Arnob Ghosh and Abhishek Gupta and Ness B. Shroff},
      year={2022},
      eprint={2207.00908},
      archivePrefix={arXiv},
      primaryClass={cs.NI}}
'''
import math
import numpy as np
from utils.posterior import posterior_update
import ipdb

class GP_UCB():
    def __init__(self, n_arms, B, kernel, Lambda, R):
        """
        @param Lambda, R: hyperparameters
        @param n_arms: number of arms 
        @param B: exploration parameter used in calculating optimal beta
        @param kernel: kernel function model the correlation between arms
        """
        self.B = B
        self.Lambda = Lambda
        self.R = R
        self.n_arms = n_arms

        self.round = 0                       # to count the number of rounds
        self.mean_vec_x = np.zeros(self.n_arms)     # posterior mean
        self.cov_mat_x = kernel
        self.var_vec_x = (1/self.Lambda) * (np.diag(self.cov_mat_x))  # posterior variance
      

    # def __str__(self):
    #     return 'UCB policy, alpha = {}'.format(self.alpha)

    def pull_arm(self):
        if self.round == 0:
            arm_pld = 7
            # arm_pld = np.random.randint(self.n_arms)
            self.round += 1
        else:
            # ipdb.set_trace()
            beta_t = self.optimal_beta_selection()
            arm_pld = np.argmax(self.mean_vec_x + beta_t * np.sqrt(self.var_vec_x))      # Alg. 1 Line 6 in ref paper
            self.round += 1      
        return arm_pld
    
    def optimal_beta_selection(self):
        """
        return: optimal beta for exploration_exploitation trade-off at each round.
        """
        gamma_t = math.log(self.round)
        return self.B + self.R/math.sqrt(self.Lambda) * math.sqrt(gamma_t)      # Alg. 1 Line 5 in ref paper
    
    def update(self, arm, reward):
        """
        @param arm: selected arm for current round
        @param reward: reward for selected arm 
        """
        self.mean_vec_x, self.cov_mat_x = posterior_update(self.mean_vec_x, self.cov_mat_x, reward, arm, self.Lambda)   # Alg. 1 Line 9 and Eq. 3-4 in ref paper
        self.var_vec_x = (1/self.Lambda) * (np.diag(self.cov_mat_x))

class GP_UCB_Constraint(GP_UCB):
    def __init__(self, n_arms, B, kernel_x, kernel_y, Lambda, R, phi_max, eta):
        """
        @param Lambda, R, phi_max, eta: hyperparameters
        @param n_arms: number of arms 
        @param B: exploration parameter used in calculating optimal beta
        @param kernel: kernel function model - the correlation between arms
        """
        super().__init__(n_arms, B, kernel_x, Lambda, R)
        self.phi_max = phi_max
        self.eta = eta

        self.round = 0                              # to count the number of rounds
        self.mean_vec_y = np.zeros(self.n_arms)     # posterior mean
        self.cov_mat_y = kernel_y
        self.var_vec_y = (1/self.Lambda) * (np.diag(self.cov_mat_y))  # posterior variance
        self.phi_t = phi_max
    
    def pull_arm(self):
        if self.round == 0:
            arm_pld = np.random.randint(self.n_arms)
            self.round += 1
            beta_t = super().optimal_beta_selection()
        else:
            # ipdb.set_trace()
            beta_t = super().optimal_beta_selection()
            f_est = self.mean_vec_x + beta_t * np.sqrt(self.var_vec_x)     # Alg. 1 Line 6 in ref paper
            g_est = self.mean_vec_y - beta_t * np.sqrt(self.var_vec_y)     # Alg. 1 Line 6 in ref paper
            arm_pld = np.argmax(f_est - self.phi_t * g_est)                # Alg. 1 Line 7 and 8 - defining acqusition and choosing beamforming vector
            self.round += 1      
        return arm_pld, beta_t
    
    def update(self, arm, reward, violation, beta_t):
        """
        @param arm: selected arm for current round
        @param reward: reward for selected arm 
        @param violation: violation for selected arm
        @param beta_t: exploration parameter 
        """
        super().update(arm, reward)
        self.mean_vec_y, self.cov_mat_y = posterior_update(self.mean_vec_y, self.cov_mat_y, violation, arm, self.Lambda)   # Alg. 1 Line 9 and Eq. 3-4 in ref paper
        self.var_vec_y = (1/self.Lambda) * (np.diag(self.cov_mat_y))

        g_est = self.mean_vec_y - beta_t * np.sqrt(self.var_vec_y)        # Alg. 1 Line 10 Updating dual
        temp = self.phi_max + self.eta * g_est[arm]                       # updating acqusition parameter self.phi_t
        if temp <= 0:
            self.phi_max = 0
        elif temp >= self.phi_max:
            self.phi_t = self.phi_max
        else:
            self.phi_t = temp

class RE_GP_UCB(GP_UCB):
    def __init__(self, n_arms, B, kernel, Lambda, R, reset_noB):
        """
        @param Lambda, R: hyperparameters
        @param n_arms: number of arms 
        @param B: exploration parameter used in calculating optimal beta
        @param kernel: kernel function model the correlation between arms
        """
        super().__init__(n_arms, B, kernel, Lambda, R)
        self.noB = reset_noB
        self.kernel = kernel
        self.reset = 0

    # def __str__(self):
    #     return 'UCB policy, alpha = {}'.format(self.alpha)

    def pull_arm(self):
        if self.round == 0:
            arm_pld = np.random.randint(self.n_arms)
            self.round += 1
            self.reset += 1
        else:
            # ipdb.set_trace()
            beta_t = self.optimal_beta_selection()
            arm_pld = np.argmax(self.mean_vec_x + beta_t * np.sqrt(self.var_vec_x))      # Alg. 1 Line 6 in ref paper
            self.round += 1    
            self.reset += 1  
        return arm_pld
    
    def optimal_beta_selection(self):
        """
        return: optimal beta for exploration_exploitation trade-off at each round.
        """
        if (self.round % self.noB) == 0:
            self.reset = 1
        gamma_t = math.log(self.reset)
        return self.B + self.R/math.sqrt(self.Lambda) * math.sqrt(gamma_t)      # Alg. 1 Line 5 in ref paper
    
    def update(self, arm, reward):
        """
        @param arm: selected arm for current round
        @param reward: reward for selected arm 
        """
        self.mean_vec_x, self.cov_mat_x = posterior_update(self.mean_vec_x, self.cov_mat_x, reward, arm, self.Lambda)   # Alg. 1 Line 9 and Eq. 3-4 in ref paper
        self.var_vec_x = (1/self.Lambda) * (np.diag(self.cov_mat_x))

        self._reset_update()
    
    def _reset_update(self):
        if not type(self.noB) == int:
            raise TypeError("`noB` must be integer") 
        
        if (self.round % self.noB) == 0:
            self.mean_vec_x = np.zeros(self.n_arms)     # posterior mean
            self.cov_mat_x = self.kernel
            self.var_vec_x = (1/self.Lambda) * (np.diag(self.cov_mat_x))

class RE_GP_UCB_Constraint(GP_UCB_Constraint):
    def __init__(self, n_arms, B, kernel_x, kernel_y, Lambda, R, phi_max, eta, reset_B):
        """
        @param Lambda, R: hyperparameters
        @param n_arms: number of arms 
        @param B: exploration parameter used in calculating optimal beta
        @param kernel: kernel function model the correlation between arms
        """
        super().__init__(n_arms, B, kernel_x, kernel_y, Lambda, R, phi_max, eta)
        self.H = reset_B
        self.kernel_x = kernel_x
        self.kernel_y = kernel_y 
        self.reset = 0

    def pull_arm(self):
        if self.round == 0:
            arm_pld = np.random.randint(self.n_arms)
            self.round += 1
            self.reset += 1
            beta_t = self.optimal_beta_selection()
        else:       
        # ipdb.set_trace()
            beta_t = self.optimal_beta_selection()
            f_est = self.mean_vec_x + beta_t * np.sqrt(self.var_vec_x)     # Alg. 1 Line 6 in ref paper
            g_est = self.mean_vec_y - beta_t * np.sqrt(self.var_vec_y)     # Alg. 1 Line 6 in ref paper
            arm_pld = np.argmax(f_est - self.phi_t * g_est)                # Alg. 1 Line 7 and 8 - defining acqusition and choosing beamforming vector
            self.round += 1  
            self.reset += 1    
        return arm_pld, beta_t
    
    def optimal_beta_selection(self):
        """
        return: optimal beta for exploration_exploitation trade-off at each round.
        """
        if (self.round % self.H) == 0:
            self.reset = 1
        gamma_t = math.log(self.reset)
        return self.B + self.R/math.sqrt(self.Lambda) * math.sqrt(gamma_t)      # Alg. 1 Line 5 in ref paper
    
    def update(self, arm, reward, violation, beta_t):
        """
        @param arm: selected arm for current round
        @param reward: reward for selected arm 
        """
        super().update(arm, reward, violation, beta_t)

        self._reset_update()
    
    def _reset_update(self):
        if not type(self.H) == int:
            raise TypeError("`H(budget)` must be integer") 
        # ipdb.set_trace()
        if (self.round % self.H) == 0:
            self.mean_vec_x = np.zeros(self.n_arms)     # posterior mean
            self.cov_mat_x = self.kernel_x
            self.var_vec_x = (1/self.Lambda) * (np.diag(self.cov_mat_x))

            self.mean_vec_y = np.zeros(self.n_arms)     # posterior mean
            self.cov_mat_y = self.kernel_y
            self.var_vec_y = (1/self.Lambda) * (np.diag(self.cov_mat_y))







    
    
    
