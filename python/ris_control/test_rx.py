from scipy.io import savemat, loadmat
from ris_serial import RISController, SERIAL_BAUDRATE
import rxIQ
import logging
import time
import numpy as np
import uhd
import ipdb
from datetime import datetime
import matplotlib.pyplot as plt

"""
Give read-write permission to Arduino ports before running code
... sudo chmod 666 /dev/ttyACM*  
"""

# ipdb.set_trace()

tx_wait_time = 0.001
log_file_name = 'ris_log.log'
ris = RISController(SERIAL_BAUDRATE, log_file_name, tx_wait_time)
ris.set_serial_ports()
print(ris.serial_ports)

ris.device_initialize()
print(ris.device_dict)
print(ris.boards)

# initialize radios
# num_samps = 50000
radio_dict = {'samp_rate': 400000,
            'center_freq': 890000000,
            'gain': [5,5],
            'sources': 2,
            'addresses': "addr0=192.168.10.2",
            'subdevs': "A:0 B:0"}
size = 2000
num_samples = 200000
chans = radio_dict['sources']
uhd_radio = rxIQ.prepare_radios(radio_dict)      # prepare radio settings


arms = np.arange(0,26,dtype=int)
reward_arms = np.zeros((chans,26))
# received_data = np.zeros((26,size))
received_data = np.zeros((chans, size*26), dtype=np.complex64) 
# Load the .mat file into a dictionary
mat_file = 'online_data/arms26.mat'
mat_data = loadmat(mat_file)
arm_list = mat_data['arms'].tolist()
experiments = 200
print(arm_list)
# weights1 = list()
# weights2 = list()
# data = create_rx_stream(uhd_radio,2000000,chans,size,True)
# time.sleep(4)
print("data collection starting...")
rewards_all = []
rx_streamer, metadata = rxIQ.setup_stream(uhd_radio, chans)

for exp in range(experiments):

    for arm in arms:
            # time.sleep(3)
        # if arm == 0:
            

        # for i in range(600):
        # print(time.time())
        # ipdb.set_trace()
        packets={0: ris.create_packet(arm_list[arm][0]), 1: ris.create_packet(arm_list[arm][1])}
        ris.start_thread(packets, 0)
        ris.start_thread(packets, 1)

        time.sleep(0.001)
        data, dt = rxIQ.receive(uhd_radio, rx_streamer, metadata, chans, size)
        for chan in range(chans):
            reward_arms[chan][arm] = np.mean(np.abs(data[chan][500:1500])) * 100               # first samples are zero
            # received_data[chan][0][arm*size:(arm+1)*size] = data[chan]
        # print(abs(data[0]))
        # rxIQ.stop_rx_stream(rx_streamer)

    # plt.figure(figsize=(12,8))
    # plt.plot(reward_arms[0], label = "receiver")
    # # plt.plot(reward_arms[1], label = "interferer")
    # plt.xlabel("Arms", fontsize=20)
    # plt.ylabel("Received Signal Strength", fontsize=20)
    # plt.legend(bbox_to_anchor=(1, 1))
    # plt.show()
    # time.sleep(1)

    savemat('results/hanus_lab/rx_data_loc17_exp'+ str(exp)+'.mat', {'rewards':reward_arms})



    
    


