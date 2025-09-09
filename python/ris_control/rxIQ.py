#!/usr/bin/env python
import uhd
# from gnuradio import gr
import time
import numpy
from scipy.io import savemat
import ipdb
import numpy as np
import matplotlib.pyplot as plt
import datetime


def set_center_freq(uhd_radio, center_freq, sources):
    # Tune all channels to the desired frequency
    
    tune_req =  uhd.libpyuhd.types.tune_request(center_freq)
    tune_resp = uhd_radio.set_rx_freq(tune_req, 0)
    print(uhd_radio.get_rx_freq(0))
    tune_req.rf_freq = center_freq
    tune_req.rf_freq_policy = uhd.libpyuhd.types.tune_request_policy.manual
    tune_req.dsp_freq_policy = uhd.libpyuhd.types.tune_request_policy.manual
    tune_req.dsp_freq = tune_resp.actual_dsp_freq

    for chan in range(sources):
            uhd_radio.set_rx_freq(tune_req, chan)

    # Synchronize the tuned channels
    now = uhd_radio.get_time_now()

    # ipdb.set_trace()
    uhd_radio.set_command_time(now + uhd.libpyuhd.types.time_spec(0.01))

    for chan in range(sources):
        uhd_radio.set_rx_freq(tune_req, chan)

    time.sleep(0.11)

    uhd_radio.clear_command_time()
    # print(uhd_radio.get_rx_freq(0))

def setup_stream(uhd_radio, chans):
    #create a receive streamer
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = range(chans)
    metadata = uhd.types.RXMetadata()
    rx_streamer = uhd_radio.get_rx_stream(st_args)
    # print(st_args)

    return rx_streamer, metadata
    
def receive(uhd_radio,rx_streamer, metadata, chans, size):
    # RX_CLEAR_COUNT = 1000
    # Create the array to hold the return samples.
    samples = np.empty((chans, size), dtype=np.complex64)

    # Figure out the size of the receive buffer and make it
    buffer_samps = rx_streamer.get_max_num_samps()

     # Receive Samples
    recv_buffer = numpy.zeros((chans, buffer_samps), dtype=numpy.complex64)   
    # rx_streamer.recv(recv_buffer, metadata)
    rx_stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    rx_stream_cmd.num_samps = size
    rx_stream_cmd.stream_now = False
    rx_stream_cmd.time_spec = uhd_radio.get_time_now() + uhd.libpyuhd.types.time_spec(0.02)
    rx_streamer.issue_stream_cmd(rx_stream_cmd)

    recv_samps = 0
    dt = datetime.datetime.now()#datetime.timezone.utc)
    while recv_samps < size:
        samps = rx_streamer.recv(recv_buffer, metadata)

        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
            print(metadata.strerror())
        if samps:
            real_samps = min(size - recv_samps, samps)
            samples[:, recv_samps:recv_samps + real_samps] = \
                recv_buffer[:, 0:real_samps]
            recv_samps += real_samps
    # Done.  Return samples.
    return samples, dt

def stop_rx_stream(rx_streamer):
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    rx_streamer.issue_stream_cmd(stream_cmd)

def create_rx_stream(uhd_radio, num_samps, chans, size, cont=False):
    #create a receive streamer
    # num_samps = 400000
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = range(chans)
    metadata = uhd.types.RXMetadata()
    rx_streamer = uhd_radio.get_rx_stream(st_args)

    #Setup streaming
    # ipdb.set_trace()
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = False
    stream_cmd.time_spec = uhd_radio.get_time_now() + uhd.libpyuhd.types.time_spec(1.0)
    rx_streamer.issue_stream_cmd(stream_cmd)
    recv_buffer = numpy.zeros((chans, size), dtype=numpy.complex64)

    # #Setup streaming
    # stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    # stream_cmd.stream_now = False
    # stream_cmd.time_spec = uhd_radio.get_time_now() + uhd.libpyuhd.types.time_spec(1.0)
    # rx_streamer.issue_stream_cmd(stream_cmd)

    # Receive Samples
    if cont == True:
        samples = numpy.zeros((chans, num_samps), dtype=numpy.complex64)   # fc64-np.complex128, fc32-np.complex64 --> Complex-valued double-precision data
        for i in range(num_samps//size):
            rx_streamer.recv(recv_buffer, metadata)
            for chan in range(chans):
                samples[chan][i*size:(i+1)*size] = recv_buffer[chan]
    else:
         samples = numpy.zeros((chans, size), dtype=numpy.complex64) 
         rx_streamer.recv(recv_buffer, metadata)
         samples = recv_buffer

    # Stop Stream
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    rx_streamer.issue_stream_cmd(stream_cmd)

    # ipdb.set_trace()
    # print(samples.shape)
    # print(samples)

    return samples

def prepare_radios(radio_dict:dict):
    uhd_usrp_source_0 = uhd.usrp.MultiUSRP(radio_dict['addresses'])
    uhd_usrp_source_0.set_clock_source('external')
    uhd_usrp_source_0.set_time_source('external')

    sources = radio_dict['sources']
    # ipdb.set_trace()
    if 'serial' in radio_dict['addresses']:
        for device in range(len(radio_dict['addresses'].split('serial'))-1):
            uhd_usrp_source_0.set_rx_subdev_spec(uhd.usrp.SubdevSpec(radio_dict['subdevs']),device)
    else:
        for device in range(len(radio_dict['addresses'].split('addr'))-1):
            uhd_usrp_source_0.set_rx_subdev_spec(uhd.usrp.SubdevSpec(radio_dict['subdevs']),device)
    # Set channel specific settings
    for chan in range(sources):
            uhd_usrp_source_0.set_rx_rate(radio_dict['samp_rate'], chan)
            uhd_usrp_source_0.set_rx_gain(radio_dict['gain'][chan], chan)
            uhd_usrp_source_0.set_rx_dc_offset(False, chan)
            uhd_usrp_source_0.set_rx_antenna('RX2', chan)
            uhd_usrp_source_0.set_rx_iq_balance(True, chan)
            

    # Reset radios' sense of time to 0.000s on the next PPS edge:
    uhd_usrp_source_0.set_time_next_pps(uhd.libpyuhd.types.time_spec(0.0))
    time.sleep(1)

    # Use timed commands to set frequencies
    #This will ensure that the LO and DSP chain of our USRPs are retuned synchronously (on the same clock cycle).
    set_center_freq(uhd_usrp_source_0, radio_dict['center_freq'], sources)

    return uhd_usrp_source_0

def remove_radio(uhd_radio):
    del uhd_radio

def get_avg_power(samps):
    return 10.0 * np.log10(np.sum(np.square(np.abs(samps)))/len(samps))

 

##################################################
if __name__ == '__main__':
    radio_dict = {'samp_rate': 400000,
                  'center_freq': 890000000,
                  'gain': [0,0],
                  'sources': 2,
                  'addresses': "addr0=192.168.10.2",    #"serial0=317229F,serial1=318D28D",   
                  'subdevs': "A:0 B:0"}

    size = 2000
    chans = radio_dict['sources']
    uhd_radio = prepare_radios(radio_dict)      # prepare radio settings
    # ipdb.set_trace()
    rx_streamer, metadata = setup_stream(uhd_radio, chans)
    print("start")
    # for i in range(0,10):
    #     ipdb.set_trace()
        
    data, dt = receive(uhd_radio, rx_streamer, metadata, chans, size)
       
        
    #     received_Data = np.mean(np.abs(data[0][20::]))                  # first 20 samples are zero
    print(data[0])
    # print(data[1])
    print(dt)
    
    # stop_rx_stream(rx_streamer)

    # data = create_rx_stream(uhd_radio,10*400000,chans,size,True)
    # data_dic = {"ant1":data[0], "ant2":data[1], "ant3":data[2], "ant4":data[3]}
    # plt.figure(figsize=(12,8))
    # plt.plot(data[0], label = "best")

    plt.figure(figsize=(12,8))
    plt.plot(data[0], label = "receiver")
    plt.plot(data[1], label = "interferer")
    plt.xlabel("Arms", fontsize=20)
    plt.ylabel("Received Signal Strength", fontsize=20)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.show()

    savemat("deneme_file.mat", {"ant1":data[0]})



