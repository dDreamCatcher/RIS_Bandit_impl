import numpy as np
from GPUCB import GP_UCB, GP_UCB_Constraint 
import data_processing as dpr
from scipy.io import loadmat, savemat
import ipdb
from time import time

def calculate_regret(rewards, arm):
    """
    @param rewards: f_test - rewards of received signal for each arm(selected beam vector)
    @param arm: pulled arm 
    :return: regret of pulled arm
    """
    return np.max(rewards) - rewards[arm]

def calculate_regret_constraint(rewards, constraint_opt, arm):
    """
    @param rewards: f_test - rewards of received signal for each arm(selected beam vector)
    @param arm: pulled arm 
    :return: regret of pulled arm
    """
    return constraint_opt - rewards[arm]

def calculate_reward(rewards, arm, epsilon):
    """
    @param rewards: f_test - rewards of received signal at target rx for each arm(selected beam vector)
    @param arm: pulled arm 
    @epsilon: hyperparameter to add noise term to reward
    :return: reward of pulled arm or best arm
    """
    return rewards[arm] + epsilon * np.random.rand()

def calculate_violation(violations, arm, epsilon):
    """
    @param violations: g_test - violations of received signal at interferer for each arm(selected beam vector)
    @param arm: pulled arm 
    @epsilon: hyperparameter to add noise term to violation
    :return: violation of pulled arm or best arm
    """
    return violations[arm] + epsilon * np.random.rand()

def find_constraint_optimum(rewards, violations, opt_value):
    """
    @param rewards: f_test - rewards of received signal at target rx for each arm(selected beam vector)
    @param violations: g_test - violations of received signal at interferer for each arm(selected beam vector)
    @param opt_value: initilized value for optimum rewards with constraint
    @return opt_value, index: return found optimum constrint reward and arm index 
    """
    for i in range(len(rewards)):
        if (violations[i] <= 0) and (rewards[i] > opt_value):
            opt_value = rewards[i]
            index = i
    return opt_value, index

def offline_evaluate(gb_ucb:GP_UCB, rewards, n_rounds=None, epsilon=0.0):
    """
    @param gb_ucb: object instance created from class GP_UCB
    @param rewards: rewards of received signal for each arm
    @param n_rounds: number of rounds to run an evaluation
    @param epsilon: hyperparameter to add noise term to reward used in function calculate_reward
    
    """
    if n_rounds == None:        # set n_rounds to infinite number to run until all data exhausted
        n_rounds = np.inf
    elif not type(n_rounds) == int:
        raise TypeError("`n_rounds` must be integer or default 'None'")
    
    pulled_arms = np.zeros(n_rounds, dtype=int )                    # array to contain chosen arms 
    best_arms = np.zeros(n_rounds, dtype=int)                      # array to contain best(expected) arms 
    reward_arms = np.zeros(n_rounds)                    # rewards of each chosen arm
    cumulative_reward = np.zeros(n_rounds)              # cumulative reward at each iteration
    average_reward = np.zeros(n_rounds)                 # average reward at each iteration 
    cumulative_regret = np.zeros(n_rounds)              # cumulative regret at each iteration    
    average_regret = np.zeros(n_rounds)                 # average regret at each iteration
    cumulative_expected_reward = np.zeros(n_rounds)     # cumulative expected reward at each iteration 
    average_expected_reward = np.zeros(n_rounds)       # average expected reward at each iteration 
  
    cum_reward = 0
    cum_reward_max = 0
    cum_regret = 0
    for round in range(n_rounds):
        # pull an arm and, calculate reward for that, and expected reward
        arm = gb_ucb.pull_arm()           
        pulled_arms[round] = arm
        best_arms[round] = np.argmax(rewards)

        reward_arms[round] = calculate_reward(rewards, arm, epsilon)
        cum_reward += reward_arms[round]
        cumulative_reward[round] = cum_reward
        average_reward[round] = cum_reward / (round+1)

        # ipdb.set_trace()
        cum_reward_max += calculate_reward(rewards, best_arms[round], epsilon)
        cumulative_expected_reward[round] = cum_reward_max
        average_expected_reward[round] = cum_reward_max / (round+1)

        #calculate regret
        cum_regret += calculate_regret(rewards, arm)
        cumulative_regret[round] = cum_regret
        average_regret[round] = cum_regret / (round+1)

        #update the kernel parameters
        gb_ucb.update(arm, reward_arms[round])

    return {'pulled_arms': pulled_arms, 'best_arms': best_arms, 'reward_arms': reward_arms, 'cumulative_reward': cumulative_reward, 
            'cumulative_expected_reward': cumulative_expected_reward, 'average_reward': average_reward, 'average_expected_reward': average_expected_reward,
             'cumulative_regret': cumulative_regret, 'average_regret': average_regret}

def offline_evaluate_contraint(gb_ucb:GP_UCB_Constraint, rewards, violations, n_rounds=None, epsilon=0.0):
    """
    @param gb_ucb: object instance created from class GP_UCB
    @param rewards: rewards of received signal for each arm
    @param violations: violations of received signal for each arm
    @param n_rounds: number of rounds to run an evaluation
    @param epsilon: hyperparameter to add noise term to reward used in function calculate_reward
    """
    if n_rounds == None:        # set n_rounds to infinite number to run until all data exhausted
        n_rounds = np.inf
    elif not type(n_rounds) == int:
        raise TypeError("`n_rounds` must be integer or default 'None'")
    
    pulled_arms = np.zeros(n_rounds, dtype=int )                    # array to contain chosen arms 
    best_arms = np.zeros(n_rounds, dtype=int)                      # array to contain best(expected) arms 
    reward_arms = np.zeros(n_rounds)                    # rewards of each chosen arm
    cumulative_reward = np.zeros(n_rounds)              # cumulative reward at each iteration
    average_reward = np.zeros(n_rounds)                 # average reward at each iteration 
    cumulative_regret = np.zeros(n_rounds)              # cumulative regret at each iteration    
    average_regret = np.zeros(n_rounds)                 # average regret at each iteration
    cumulative_expected_reward = np.zeros(n_rounds)     # cumulative expected reward at each iteration 
    average_expected_reward = np.zeros(n_rounds)       # average expected reward at each iteration 
    violation_arms = np.zeros(n_rounds)                 # violation of each chosen arm
    cumulative_violation = np.zeros(n_rounds)           # cumulative violation at each iteration 
    average_violation = np.zeros(n_rounds)              # average violation at each iteration
  
    cum_reward = 0
    cum_reward_max = 0
    cum_regret = 0
    cum_violation = 0
    for round in range(n_rounds):
        # pull an arm and, calculate reward for that, and expected reward
        opt_value = -10
        arm, beta = gb_ucb.pull_arm()           
        pulled_arms[round] = arm
        const_opt, best_arms[round] = find_constraint_optimum(rewards, violations, opt_value)

        reward_arms[round] = calculate_reward(rewards, arm, epsilon)
        cum_reward += reward_arms[round]
        cumulative_reward[round] = cum_reward
        average_reward[round] = cum_reward / (round+1)

        # ipdb.set_trace()
        cum_reward_max += calculate_reward(rewards, best_arms[round], epsilon)          # constraint expected reward
        cumulative_expected_reward[round] = cum_reward_max
        average_expected_reward[round] = cum_reward_max / (round+1)

        #calculate violation
        violation_arms[round] = calculate_violation(violations, arm, epsilon)
        cum_violation += violation_arms[round]
        cumulative_violation[round] = cum_violation
        average_violation[round] = cum_violation / (round+1)

        #calculate regret
        cum_regret += calculate_regret_constraint(rewards, const_opt, arm)
        cumulative_regret[round] = cum_regret
        average_regret[round] = cum_regret / (round+1)

        #update the kernel parameters
        gb_ucb.update(arm, reward_arms[round], violation_arms[round], beta)

    return {'pulled_arms': pulled_arms, 'best_arms': best_arms, 'reward_arms': reward_arms, 'cumulative_reward': cumulative_reward, 
            'cumulative_expected_reward': cumulative_expected_reward, 'average_reward': average_reward, 'average_expected_reward': average_expected_reward,
             'cumulative_regret': cumulative_regret, 'average_regret': average_regret, 'violation_arms': violation_arms, 'cumulative_violation': cumulative_violation,
             'average_violation': average_violation}


if __name__=="__main__":
    t0 = time()
    np.random.seed(1337)
    dpr.main()              # data processing
    save_file = False

    dict_rewards = loadmat('data/rewards.mat')
    rewards = dict_rewards['rewards'][0]
    n_arms = len(rewards)
    B = max(abs(rewards))
    dict_kernel = loadmat('data/se_kernel.mat') 
    kernel = dict_kernel['kernel_x']
    Lambda = 1
    R = 1

    gp_ucb = GP_UCB(n_arms, B, kernel, Lambda, R)

    results = offline_evaluate(gp_ucb, rewards, 1000, 0.0)
    t1 = time()

    # print('it takes %f', (t1-t0))
    if save_file:
        savemat('results/results.mat', results)



                                                                         





