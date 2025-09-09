import numpy as np
from GPUCB import GP_UCB, GP_UCB_Constraint, RE_GP_UCB, RE_GP_UCB_Constraint
import data_processing as dpr
from scipy.io import loadmat, savemat
import ipdb
import os
from time import time, sleep
from ris_control.ris_serial import RISController, SERIAL_BAUDRATE
import ris_control.rxIQ as rxIQ
import matplotlib.pyplot as plt

# def file_mat2list(path_name, filename, key):
#     mat_file = os.path.join(path_name,filename)
#     mat_data = loadmat(mat_file)

def initialize(radio_dict):
    # initialize serial communication between RIS and basestation
    tx_wait_time = 0.01
    log_file_name = 'ris_log.log'
    ris = RISController(SERIAL_BAUDRATE, log_file_name, tx_wait_time)
    ris.set_serial_ports()
    print(ris.serial_ports)

    # initialize devices
    ris.device_initialize()
    print(ris.device_dict)
    print(ris.boards)

    # initialize radios
    # num_samps = 50000
    chans = radio_dict['sources']
    uhd_radio = rxIQ.prepare_radios(radio_dict)      # prepare radio settings

    return chans, uhd_radio, ris

#     return mat_data['key'].tolist()
def run_alg(gb_ucb:GP_UCB, arm_list, uhd_radio, chans, ris:RISController, n_rounds=None):
    if n_rounds == None:        # set n_rounds to infinite number to run until all data exhausted
        n_rounds = np.inf
    elif not type(n_rounds) == int:
        raise TypeError("`n_rounds` must be integer or default 'None'")
    
    pulled_arms = np.zeros(n_rounds, dtype=int )                    # array to contain chosen arms 
    reward_arms = np.zeros(n_rounds)                    # rewards of each chosen arm
    cumulative_reward = np.zeros(n_rounds)              # cumulative reward at each iteration
    average_reward = np.zeros(n_rounds)                 # average reward at each iteration 
    size = 2000

    cum_reward = 0
    for round in range(n_rounds):
        # pull an arm and save it
        arm = gb_ucb.pull_arm()    
        # arm = round       
        pulled_arms[round] = arm

        if round == 0:
            rx_streamer, metadata = rxIQ.setup_stream(uhd_radio, chans)

        # ipdb.set_trace()
        # switch the config of RIS according to the pulled arm
        packets={0: ris.create_packet(arm_list[arm][0]), 1: ris.create_packet(arm_list[arm][1])}   #create packets to send them to devices
        ris.start_thread(packets, 0)                                                               # send to device 0
        ris.start_thread(packets, 1)                                                               # send to device 1
        sleep(0.01)

        # record received signal
        # logging.debug(f'Receiving IQs started at {time.time()}')
        data, dt = rxIQ.receive(rx_streamer, metadata, chans, size)
        reward_arms[round] = np.mean(np.abs(data[0][500:1500])) * 100               # first samples are zero

        cum_reward += reward_arms[round]
        cumulative_reward[round] = cum_reward
        average_reward[round] = cum_reward / (round+1)

        #update the kernel parameters
        gb_ucb.update(arm, reward_arms[round])

    return {'pulled_arms': pulled_arms, 'reward_arms': reward_arms, 'average_reward': average_reward}

def run_alg_const(gb_ucb:GP_UCB, arm_list, uhd_radio, chans, ris:RISController, n_rounds=None):
    if n_rounds == None:        # set n_rounds to infinite number to run until all data exhausted
        n_rounds = np.inf
    elif not type(n_rounds) == int:
        raise TypeError("`n_rounds` must be integer or default 'None'")
    
    pulled_arms = np.zeros(n_rounds, dtype=int )                    # array to contain chosen arms 
    reward_arms = np.zeros(n_rounds)                    # rewards of each chosen arm
    cumulative_reward = np.zeros(n_rounds)              # cumulative reward at each iteration
    average_reward = np.zeros(n_rounds)                 # average reward at each iteration 
    violation_arms = np.zeros(n_rounds)                 # violation of each chosen arm
    cumulative_violation = np.zeros(n_rounds)           # cumulative violation at each iteration 
    average_violation = np.zeros(n_rounds)              # average violation at each iteration 
    size = 2000

    cum_reward = 0
    cum_violation = 0
    for round in range(n_rounds):
        # pull an arm and save it
        arm, beta = gb_ucb.pull_arm()    
        # arm = round       
        pulled_arms[round] = arm

        if round == 0:
            rx_streamer, metadata = rxIQ.setup_stream(uhd_radio, chans)

        # ipdb.set_trace()
        # switch the config of RIS according to the pulled arm
        packets={0: ris.create_packet(arm_list[arm][0]), 1: ris.create_packet(arm_list[arm][1])}   #create packets to send them to devices
        ris.start_thread(packets, 0)                                                               # send to device 0
        ris.start_thread(packets, 1)                                                               # send to device 1
        sleep(0.01)

        # record received signal
        data, dt = rxIQ.receive(uhd_radio, rx_streamer, metadata, chans, size)
        reward_arms[round] = np.mean(np.abs(data[0][500:1500])) * 100               # first samples are zero
        violation_arms[round] = np.mean(np.abs(data[1][500:1500])) * 100  
          
        # bunlari sonra hesaplamak daha iyi
        cum_reward += reward_arms[round]
        cumulative_reward[round] = cum_reward
        average_reward[round] = cum_reward / (round+1)
        # bunlari sonra hesaplamak daha iyi
        cum_violation += violation_arms[round]
        cumulative_violation[round] = cum_violation
        average_violation[round] = cum_violation / (round+1)

        #update the kernel parameters
        gb_ucb.update(arm, reward_arms[round], violation_arms[round], beta)

    return {'pulled_arms': pulled_arms, 'reward_arms': reward_arms, 'average_reward': average_reward,
            'violation_arms': violation_arms, 'average_violation': average_violation}


def online_evaluate(gb_ucb:GP_UCB, arm_list, radio_dict, n_rounds=None):
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
    reward_arms = np.zeros(n_rounds)                    # rewards of each chosen arm
    cumulative_reward = np.zeros(n_rounds)              # cumulative reward at each iteration
    average_reward = np.zeros(n_rounds)                 # average reward at each iteration 
  
    # ipdb.set_trace()
    # initialize serial communication between RIS and basestation
    tx_wait_time = 0.001
    log_file_name = 'ris_log.log'
    ris = RISController(SERIAL_BAUDRATE, log_file_name, tx_wait_time)
    ris.set_serial_ports()
    print(ris.serial_ports)

    # initialize devices
    ris.device_initialize()
    print(ris.device_dict)
    print(ris.boards)

    # initialize radios
    # num_samps = 50000
    size = 2000
    chans = radio_dict['sources']
    received_data = np.zeros((n_rounds,size), dtype=np.complex64)
    uhd_radio = rxIQ.prepare_radios(radio_dict)      # prepare radio settings
    print("experiment starting")
    
    cum_reward = 0
    t0 = time()
    for round in range(n_rounds):
        # pull an arm and save it
       
        arm = gb_ucb.pull_arm()    
        # arm = round       
        pulled_arms[round] = arm

        if round == 0:
            rx_streamer, metadata = rxIQ.setup_stream(uhd_radio, chans)

        # ipdb.set_trace()
        # switch the config of RIS according to the pulled arm
        packets={0: ris.create_packet(arm_list[arm][0]), 1: ris.create_packet(arm_list[arm][1])}   #create packets to send them to devices
        ris.start_thread(packets, 0)                                                               # send to device 0
        ris.start_thread(packets, 1)                                                               # send to device 1
        sleep(0.001)

        # record received signal
        # logging.debug(f'Receiving IQs started at {time.time()}')
        # t2=time()
        data, dt = rxIQ.receive(uhd_radio, rx_streamer, metadata, chans, size)
        reward_arms[round] = np.mean(np.abs(data[0][500:1500])) * 100               # first samples are zero
        received_data[round] = data[0]
        # t3 = time()
        # print(t3-t2)
        # if round == 500:
        #     print("seeping")
        #     sleep(20)
        #     print("start")

        # bunlari sonra hesaplamak daha iyi
        cum_reward += reward_arms[round]
        cumulative_reward[round] = cum_reward
        average_reward[round] = cum_reward / (round+1)

        #update the kernel parameters
        gb_ucb.update(arm, reward_arms[round])
    
    t1=time()
    print(t1-t0)
    return {'pulled_arms': pulled_arms, 'reward_arms': reward_arms, 'cumulative_reward': cumulative_reward, 'average_reward': average_reward, 'received_data': received_data}

def online_evaluate_constraint(gb_ucb:GP_UCB_Constraint, arm_list, radio_dict, n_rounds=None):
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
    reward_arms = np.zeros(n_rounds)                    # rewards of each chosen arm
    cumulative_reward = np.zeros(n_rounds)              # cumulative reward at each iteration
    average_reward = np.zeros(n_rounds)                 # average reward at each iteration 
    violation_arms = np.zeros(n_rounds)                 # violation of each chosen arm
    cumulative_violation = np.zeros(n_rounds)           # cumulative violation at each iteration 
    average_violation = np.zeros(n_rounds)              # average violation at each iteration

    # initialize serial communication between RIS and basestation
    tx_wait_time = 0.001
    log_file_name = 'ris_log.log'
    ris = RISController(SERIAL_BAUDRATE, log_file_name, tx_wait_time)
    ris.set_serial_ports()
    print(ris.serial_ports)

    # initialize devices
    ris.device_initialize()
    print(ris.device_dict)
    print(ris.boards)

    # initialize radios
    # num_samps = 50000
    size = 2000
    chans = radio_dict['sources']
    received_data = np.zeros((n_rounds,size), dtype=np.complex64)
    interferer_data = np.zeros((n_rounds,size), dtype=np.complex64)
    uhd_radio = rxIQ.prepare_radios(radio_dict)      # prepare radio settings
  
    cum_reward = 0
    cum_violation = 0
    print("experiment starting")
    t0 = time()
    for round in range(n_rounds):
        # pull an arm and, calculate reward for that, and expected reward
        arm, beta = gb_ucb.pull_arm()           
        pulled_arms[round] = arm

        if round == 0:
            rx_streamer, metadata = rxIQ.setup_stream(uhd_radio, chans)
        
        # switch the config of RIS according to the pulled arm
        packets={0: ris.create_packet(arm_list[arm][0]), 1: ris.create_packet(arm_list[arm][1])}   #create packets to send them to devices
        ris.start_thread(packets, 0)                                                               # send to device 0
        ris.start_thread(packets, 1)                                                               # send to device 1
        sleep(0.001)

        # record received signal
        # logging.debug(f'Receiving IQs started at {time.time()}')
        data, dt = rxIQ.receive(uhd_radio, rx_streamer, metadata, chans, size)
        reward_arms[round] = np.mean(np.abs(data[0][500:1500])) * 100               # first samples are zero
        violation_arms[round] = np.mean(np.abs(data[1][500:1500])) * 100  
        received_data[round] = data[0]
        interferer_data[round] = data[1]

        # bunlari sonra hesaplamak daha iyi
        cum_reward += reward_arms[round]
        cumulative_reward[round] = cum_reward
        average_reward[round] = cum_reward / (round+1)
        # bunlari sonra hesaplamak daha iyi
        cum_violation += violation_arms[round]
        cumulative_violation[round] = cum_violation
        average_violation[round] = cum_violation / (round+1)

        #update the kernel parameters
        gb_ucb.update(arm, reward_arms[round], violation_arms[round], beta)
    t1 = time()
    print(t1-t0)
    return {'pulled_arms': pulled_arms, 'reward_arms': reward_arms, 'cumulative_reward': cumulative_reward, 'average_reward': average_reward, 
            'received_data': received_data, 'violation_arms': violation_arms, 'cumulative_violation': cumulative_violation, 
            'average_violation': average_violation, 'interferer_data': interferer_data}



if __name__=="__main__":
    save_file = True
    scenarios = ["Static", "Static_constraint", "Varying", "Varying_constraint"]
    scenario = scenarios[2]

    if scenario == "Static":

        radio_dict = {'samp_rate': 400000,
                    'center_freq': 890000000,
                    'gain': [5],
                    'sources': 1,
                    'addresses': "addr0=192.168.50.2",
                    'subdevs': "A:0"}
        
        
        # np.random.seed(1337)
        arm_data = loadmat('online_data/arms26.mat')
        arm_list = arm_data['arms'].tolist()
        n_arms = len(arm_list[0:12])
        B = 46
        dict_kernel = loadmat('data/half_m_kernel.mat') 
        kernel = dict_kernel['kernel_x']
        Lambda = 6.25
        R = 0.25

        gp_ucb = GP_UCB(n_arms, B, kernel, Lambda, R)

        t0 = time()
        results = online_evaluate(gp_ucb, arm_list, radio_dict, 1000)
        t1 = time()

        plt.figure(figsize=(12,8))
        plt.plot(results['average_reward'], label = "best")

        plt.figure(figsize=(12,8))
        plt.plot(results['reward_arms'], label = "best")
        plt.show()

        print(results['pulled_arms'])

        print('it takes %f', (t1-t0))
        if save_file:
            savemat('results/static/results_online_high.mat', results)

    elif scenario == "Static_constraint":
        radio_dict = {'samp_rate': 400000,
                    'center_freq': 890000000,
                    'gain': [5,2],
                    'sources': 2,
                    'addresses': "addr0=192.168.50.2",
                    'subdevs': "A:0 B:0"}
        # np.random.seed(1337)
        arm_data = loadmat('online_data/arms26.mat')
        arm_list = arm_data['arms'].tolist()
        n_arms = len(arm_list)
        B = 40.5
        dict_kernel = loadmat('data/m_kernel.mat') 
        kernel_x = dict_kernel['kernel_x']
        kernel_y = dict_kernel['kernel_y']
        Lambda = 0.2
        R = -5
        phi_max = 25
        eta = 22

        gp_ucb = GP_UCB_Constraint(n_arms, B, kernel_x, kernel_y, Lambda, R, phi_max, eta)

        t0 = time()
        results = online_evaluate_constraint(gp_ucb, arm_list, radio_dict, 2000)
        t1 = time()

        plt.figure(figsize=(12,8))
        plt.plot(results['average_reward'], label = "best")
        plt.plot(results['average_violation'], label = "best")
        plt.figure(figsize=(12,8))
        plt.plot(results['reward_arms'], label = "best")
        plt.plot(results['violation_arms'], label = "best")
        plt.show()

        print(results['pulled_arms'])

        print('it takes %f', (t1-t0))
        if save_file:
            savemat('results/static_constraint/results_online_loc2_best_m1.mat', results)
    
    elif scenario == "Varying":

        radio_dict = {'samp_rate': 400000,
                    'center_freq': 890000000,
                    'gain': [5],
                    'sources': 1,
                    'addresses': "addr0=192.168.50.2",
                    'subdevs': "A:0"}

        # ipdb.set_trace()
        # np.random.seed(1337)
        arm_data = loadmat('online_data/arms26.mat')
        arm_list = arm_data['arms'].tolist()
        n_arms = len(arm_list[0:12])
        B = 60
        dict_kernel = loadmat('data/half_m_kernel.mat') 
        kernel = dict_kernel['kernel_x']
        Lambda = 6.25
        R = 0.25
        reset = 200

        gp_ucb = RE_GP_UCB(n_arms, B, kernel, Lambda, R, reset)
        # sleep(5)

        t0 = time()
        results = online_evaluate(gp_ucb, arm_list, radio_dict, 1000)
        t1 = time()

        plt.figure(figsize=(12,8))
        plt.plot(results['average_reward'], label = "best")

        plt.figure(figsize=(12,8))
        plt.plot(results['reward_arms'], label = "best")
        plt.show()

        print(results['pulled_arms'])

        print('it takes %f', (t1-t0))
        if save_file:
            savemat('results/varying/results4_online_speed90_B400.mat', results)
    
    elif scenario == "Varying_constraint":

        radio_dict = {'samp_rate': 400000,
                    'center_freq': 890000000,
                    'gain': [5,5],
                    'sources': 2,
                    'addresses': "addr0=192.168.50.2",
                    'subdevs': "A:0 B:0"}
        
        # np.random.seed(1337)
        arm_data = loadmat('online_data/arms26.mat')
        arm_list = arm_data['arms'].tolist()
        n_arms = len(arm_list)
        B = 50
        dict_kernel = loadmat('data/se_kernel.mat') 
        kernel_x = dict_kernel['kernel_x']
        kernel_y = dict_kernel['kernel_y']
        Lambda = 0.2
        R = -5
        phi_max = 25
        eta = 22
        
        reset = 500

        gp_ucb = RE_GP_UCB_Constraint(n_arms, B, kernel_x, kernel_y, Lambda, R, phi_max, eta, reset)
        # sleep(5)

        t0 = time()
        results = online_evaluate_constraint(gp_ucb, arm_list, radio_dict, 1000)
        t1 = time()

        plt.figure(figsize=(12,8))
        plt.plot(results['average_reward'], label = "best")

        plt.figure(figsize=(12,8))
        plt.plot(results['average_violation'], label = "best")
        # plt.show()

        plt.figure(figsize=(12,8))
        plt.plot(results['reward_arms'], label = "best")
        plt.show()

        print(results['pulled_arms'])

        print('it takes %f', (t1-t0))
        if save_file:
            savemat('results/varying/results_online_500.mat', results)




                                                                         





