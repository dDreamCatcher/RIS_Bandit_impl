import numpy as np
from GPUCB import GP_UCB, GP_UCB_Constraint, RE_GP_UCB_Constraint
import data_processing as dpr
from scipy.io import loadmat, savemat
import ipdb
from time import time
import math
import matplotlib.pyplot as plt

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

def calculate_violation(violations, arm, epsilon, int_th):
    """
    @param violations: g_test - violations of received signal at interferer for each arm(selected beam vector)
    @param arm: pulled arm 
    @epsilon: hyperparameter to add noise term to violation
    @int_th: threshold value for the constraint 
    :return: violation of pulled arm or best arm
    """
    return violations[arm] - int_th + epsilon * np.random.rand()

def find_constraint_optimum(rewards, violations, opt_value, int_th):
    """
    @param rewards: f_test - rewards of received signal at target rx for each arm(selected beam vector)
    @param violations: g_test - violations of received signal at interferer for each arm(selected beam vector)
    @param opt_value: initilized value for optimum rewards with constraint
    @return opt_value, index: return found optimum constrint reward and arm index 
    """
    for i in range(len(rewards)):
        if (violations[i] <= int_th) and (rewards[i] > opt_value):
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
        # ipdb.set_trace()
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

def offline_evaluate_contraint(gb_ucb:GP_UCB_Constraint, rewards, violations, int_th=0.0, n_rounds=None, epsilon=0.0):
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
        const_opt, best_arms[round] = find_constraint_optimum(rewards, violations, opt_value, int_th)

        reward_arms[round] = calculate_reward(rewards, arm, epsilon)
        cum_reward += reward_arms[round]
        cumulative_reward[round] = cum_reward
        average_reward[round] = cum_reward / (round+1)

        # ipdb.set_trace()
        cum_reward_max += calculate_reward(rewards, best_arms[round], epsilon)          # constraint expected reward
        cumulative_expected_reward[round] = cum_reward_max
        average_expected_reward[round] = cum_reward_max / (round+1)

        #calculate violation
        violation_arms[round] = calculate_violation(violations, arm, epsilon, int_th)
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

def offline_evaluate_constraint_RE(gb_ucb:GP_UCB_Constraint, rewards, violations, reset, int_th=0.0, n_rounds=None, epsilon=0.0):
    """
    @param gb_ucb: object instance created from class GP_UCB
    @param rewards: rewards of received signal for each arm
    @param violations: violations of received signal for each arm
    @param n_rounds: number of rounds to run an evaluation
    @param epsilon: hyperparameter to add noise term to reward used in function calculate_reward
    @param reset: restart period W - with or withour budget
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
    best_reward_arm = np.zeros(n_rounds)  
  
    cum_reward = 0
    cum_reward_max = 0
    cum_regret = 0
    cum_violation = 0
    rewards_c = rewards[0]
    violations_c = violations[0]
    count = 0
    t0 = time()
    for round in range(n_rounds):
        # ipdb.set_trace()
        if ((round+1) % reset) == 0:
            count += 1
            if count < rewards.shape[0]:
                rewards_c = rewards[count]
                violations_c = violations[count]
            
        # pull an arm and, calculate reward for that, and expected reward
        opt_value = -10
        arm, beta = gb_ucb.pull_arm()           
        pulled_arms[round] = arm
        const_opt, best_arms[round] = find_constraint_optimum(rewards_c, violations_c, opt_value, int_th)

        reward_arms[round] = calculate_reward(rewards_c, arm, epsilon)
        cum_reward += reward_arms[round]
        cumulative_reward[round] = cum_reward
        average_reward[round] = cum_reward / (round+1)

        # ipdb.set_trace()
        best_reward_arm[round] = calculate_reward(rewards_c, best_arms[round], epsilon)  
        cum_reward_max += calculate_reward(rewards_c, best_arms[round], epsilon)          # constraint expected reward
        cumulative_expected_reward[round] = cum_reward_max
        average_expected_reward[round] = cum_reward_max / (round+1)

        #calculate violation
        violation_arms[round] = calculate_violation(violations_c, arm, epsilon, int_th)
        cum_violation += violation_arms[round]
        cumulative_violation[round] = cum_violation
        average_violation[round] = cum_violation / (round+1)

        #calculate regret
        cum_regret += calculate_regret_constraint(rewards_c, const_opt, arm)
        cumulative_regret[round] = cum_regret
        average_regret[round] = cum_regret / (round+1)

        #update the kernel parameters
        gb_ucb.update(arm, reward_arms[round], violation_arms[round], beta)
    t1 =time()
    # print(t1-t0)

    return {'pulled_arms': pulled_arms, 'best_arms': best_arms, 'reward_arms': reward_arms, 'cumulative_reward': cumulative_reward, 
            'cumulative_expected_reward': cumulative_expected_reward, 'average_reward': average_reward, 'average_expected_reward': average_expected_reward,
             'cumulative_regret': cumulative_regret, 'average_regret': average_regret, 'violation_arms': violation_arms, 'cumulative_violation': cumulative_violation,
             'average_violation': average_violation, 'best_reward_arm': best_reward_arm}

if __name__=="__main__":

    save_file = False
    scenarios = ["Static", "Static_constraint", "Varying", "Varying_constraint", "Varying_constraint_fast"]
    scenario = scenarios[3]
    # np.random.seed(1337)

    if scenario == "Static":
    
        # np.random.seed(1337)
        # dpr.main()              # data processing

        dict_rewards = loadmat('data/rewards_mobile_fast2.mat')
        # ipdb.set_trace()
        rewards = dict_rewards['rewards'][0]
        n_arms = len(rewards)
        B = max(abs(rewards))
        # B = 29.2
        dict_kernel = loadmat('data/half_se_kernel.mat') 
        kernel = dict_kernel['kernel_x']
        Lambda = 5.5
        R = 3.75

        gp_ucb = GP_UCB(n_arms, B, kernel, Lambda, R)

        results = offline_evaluate(gp_ucb, rewards, 500, 0.0)
        print(results['pulled_arms'])

        plt.figure(figsize=(12,8))
        plt.plot(rewards, linewidth=4, label = "desired")
        plt.xlabel('arms', fontsize=20)
        plt.ylabel('Received signal', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.figure(figsize=(12,8))
        plt.plot(results['average_reward'], linewidth=4, label = "alg")
        plt.plot(results['average_expected_reward'], linewidth=4, label = "desired")
        plt.xlabel('arms', fontsize=20)
        plt.ylabel('Time-average Reward', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.show()

        # print('it takes %f', (t1-t0))
        if save_file:
            savemat('results/results.mat', results)

    elif scenario == "Static_constraint":
        np.random.seed(1337)
        # dpr.main()              # data processing
        save_file = False

        dict_rewards = loadmat('data/rewards_mobile_fast2.mat')
        rewards = dict_rewards['rewards'][1]
        violations = dict_rewards['violations'][1]
        n_arms = len(rewards)
        B = [max(abs(violations))]
        # B = 33.2
        dict_kernel = loadmat('data/half_m_kernel.mat') 
        kernel_x = dict_kernel['kernel_x']
        kernel_y = dict_kernel['kernel_y']
        Lambda = [7.2]
        R = [1.75]
        phi_max = 0
        eta = 0
        th_val = 33

        # gp_ucb = GP_UCB(n_arms, B, kernel, Lambda, R)
        gp_ucb = GP_UCB_Constraint(n_arms, B, kernel_x, kernel_y, Lambda, R, phi_max, eta)

        results = offline_evaluate_contraint(gp_ucb, rewards, violations, th_val, 200, 0.0)
        t1 = time()
        print(results['pulled_arms'])

        plt.figure(figsize=(12,8))
        plt.plot(rewards, linewidth=4, label = "desired")
        # plt.plot(rewards[1], linewidth=4, label = "desired")
        plt.plot(violations, linewidth=4, label = "interferer")
        # plt.plot(violations[1], linewidth=4, label = "interferer")
        plt.legend(loc='lower right', bbox_to_anchor=(1, 1))
        plt.xlabel('arms', fontsize=20)
        plt.ylabel('Received signal', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.figure(figsize=(12,8))
        plt.plot(results['average_reward'], linewidth=4, label = "desired")
        plt.plot(results['average_expected_reward'], linewidth=4, label = "desired")
        plt.legend(loc='lower right', bbox_to_anchor=(1, 1))
        plt.xlabel('Round', fontsize=20)
        plt.ylabel('Time-Average Rewards', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.figure(figsize=(12,8))
        plt.plot(results['average_regret'], linewidth=4)
        plt.legend(loc='lower right', bbox_to_anchor=(1, 1))
        plt.xlabel("Rounds", fontsize=20)
        plt.ylabel("Average Total Regret", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.figure(figsize=(12,8))
        plt.plot(results['average_violation'], linewidth=4)
        plt.legend(loc='lower right', bbox_to_anchor=(1, 1))
        plt.xlabel("Rounds", fontsize=20)
        plt.ylabel("Average Total Violation", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.show()

        # print('it takes %f', (t1-t0))
        if save_file:
            savemat('results/results.mat', results)

    elif scenario == "Varying_constraint":

        dict_rewards = loadmat('data/rewards_mobile_fast2.mat')
        # ipdb.set_trace()
        # rewards = np.zeros((2,12))
        # violations = np.zeros((2,12))
        rewards = dict_rewards['rewards'][0:4]
        violations = dict_rewards['violations'][0:4]
        n_arms = rewards.shape[1]
        # ipdb.set_trace()
        # B = max(abs(np.ndarray.flatten(rewards)))
        B =  [max(abs(rewards[0])), max(abs(violations[1])), max(abs(violations[3]))]
        # B = max(abs(np.ndarray.flatten(violations)))
        dict_kernel = loadmat('data/half_m_kernel.mat') 
        kernel_x = dict_kernel['kernel_x']
        kernel_y = dict_kernel['kernel_y']
        Lambda = [6.15, 7.2]
        R = [1.75, 1.75]
        phi_max = 0
        eta = 0
        reset_B = 500
        reset = 200
        th_val = 31

        # gp_ucb = GP_UCB(n_arms, B, kernel, Lambda, R)
        gp_ucb = GP_UCB_Constraint(n_arms, B, kernel_x, kernel_y, Lambda, R, phi_max, eta)
        # gp_ucb = RE_GP_UCB_Constraint(n_arms, B, kernel_x, kernel_y, Lambda, R, phi_max, eta, reset_B)

        # results = offline_evaluate_contraint(gp_ucb, rewards, violations, 1000, 0.0)
        results = offline_evaluate_constraint_RE(gp_ucb, rewards, violations, reset, th_val, 800, 0.0)
        # t1 = time()
        print(results['pulled_arms'][0:1000])
        print(results['pulled_arms'][1000:1600])
        print(results['best_arms'])
        # print(results['violation_arms'])

        plt.figure(figsize=(12,8))
        plt.plot(rewards[0], linewidth=4, label = "desired")
        plt.plot(rewards[1], linewidth=4, label = "desired")
        plt.plot(violations[0], linewidth=4, label = "interferer")
        plt.plot(violations[1], linewidth=4, label = "interferer")
        plt.legend(loc='lower right', bbox_to_anchor=(1, 1))
        plt.xlabel('arms', fontsize=20)
        plt.ylabel('Received signal', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.figure(figsize=(12,8))
        plt.plot(results['average_reward'], linewidth=4, label = "averag_reward")
        plt.plot(results['average_expected_reward'], linewidth=4, label = "average_expected_reward")
        plt.legend(loc='lower right', bbox_to_anchor=(1, 1), fontsize=20)
        plt.xlabel('Rounds', fontsize=20)
        plt.ylabel('Time-Average Rewards', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.figure(figsize=(12,8))
        plt.plot(results['average_regret'], linewidth=4)
        plt.legend(loc='lower right', bbox_to_anchor=(1, 1))
        plt.xlabel("Rounds", fontsize=20)
        plt.ylabel("Time-Average Regret", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.figure(figsize=(12,8))
        plt.plot(results['average_violation'], linewidth=4)
        plt.legend(loc='lower right', bbox_to_anchor=(1, 1))
        plt.xlabel("Rounds", fontsize=20)
        plt.ylabel("Time-Average Violation", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.show()

        # print('it takes %f', (t1-t0))
        if save_file:
            savemat('results/varying/results_mobile_fast_const2.mat', results)

    elif scenario == "Varying_constraint_fast":

        dict_rewards = loadmat('data/rewards_mobile_fast2.mat')
        # ipdb.set_trace()
        rewards = dict_rewards['rewards']
        violations = dict_rewards['violations']
        # rewards = np.zeros((5,12))
        # violations = np.zeros((5,12))
        # for i in range(5):
        #     rewards[i] = dict_rewards['rewards'][i][0:12]
        #     violations[i] = dict_rewards['violations'][i][0:12]
        n_arms = rewards.shape[1]
        # ipdb.set_trace()
        # B = max(abs(np.ndarray.flatten(rewards)))
        B = 39.4
        G = max(abs(np.ndarray.flatten(violations)))
        dict_kernel = loadmat('data/half_m_kernel.mat') 
        kernel_x = dict_kernel['kernel_x']
        kernel_y = dict_kernel['kernel_y']
        Lambda = 4.25
        R = 1.25
        phi_max = 1
        eta = 8
        reset_B = 500
        reset = 400
        th_value = 33.0
        T = 2000

        # gp_ucb = GP_UCB(n_arms, B, kernel, Lambda, R)
        # gp_ucb = GP_UCB_Constraint(n_arms, B, kernel_x, kernel_y, Lambda, R, phi_max, eta)
        gp_ucb = RE_GP_UCB_Constraint(n_arms, B, kernel_x, kernel_y, Lambda, R, phi_max, eta, reset_B)

        # results = offline_evaluate_contraint(gp_ucb, rewards, violations, 1000, 0.0)
        results = offline_evaluate_constraint_RE(gp_ucb, rewards, violations, reset, th_value, T, 0.0)
        # t1 = time()
        print(results['pulled_arms'])
        print(results['best_arms'])
        # print(results['violation_arms'])
        
        for i in range(4):
            plt.figure(figsize=(12,8))
            plt.plot(rewards[i], linewidth=4, label = "desired")
            plt.plot(violations[i], linewidth=4, label = "interferer")
        plt.legend(loc='lower right', bbox_to_anchor=(1, 1))
        plt.xlabel('arms', fontsize=20)
        plt.ylabel('Received signal', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.figure(figsize=(12,8))
        plt.plot(results['average_reward'], linewidth=4, label = "averag_reward")
        plt.plot(results['average_expected_reward'], linewidth=4, label = "average_expected_reward")
        plt.legend(loc='lower right', bbox_to_anchor=(1, 1))
        plt.xlabel('Round', fontsize=20)
        plt.ylabel('Time-Average Rewards', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.figure(figsize=(12,8))
        plt.plot(results['average_regret'], linewidth=4)
        plt.legend(loc='lower right', bbox_to_anchor=(1, 1))
        plt.xlabel("Rounds", fontsize=20)
        plt.ylabel("Average Total Regret", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.figure(figsize=(12,8))
        plt.plot(results['average_violation'], linewidth=4)
        plt.legend(loc='lower right', bbox_to_anchor=(1, 1))
        plt.xlabel("Rounds", fontsize=20)
        plt.ylabel("Average Total Violation", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.show()

        # print('it takes %f', (t1-t0))
        if save_file:
            savemat('results/results_mobile_slow_const.mat', results)

                                                                         





