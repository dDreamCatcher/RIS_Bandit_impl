import serial
import serial.tools.list_ports
import logging
from scipy.io import savemat, loadmat
import sys
import glob
import traceback
import time
import threading
import ipdb
import os
import struct

"Implement serial communication between python and arduino"

SERIAL_BAUDRATE = 115200
LOG_FILE        = 'ris_log.log'
SERIAL_TX_WAIT_TIME = 0.001

class RISController:
    """RIS control module that communicates with control units (arduinos)."""
    def __init__(self, baud_rate=SERIAL_BAUDRATE, file_name = LOG_FILE, tx_wait = SERIAL_TX_WAIT_TIME):
        print('Initializing RIS Control...')

        logging.basicConfig(filename=file_name, level=logging.DEBUG)

        # print("Serial ports found:", comports)
        # self.num_serial_ports = len(self.serial_ports)
        self.tx_wait_time = tx_wait
        self.baud_rate = baud_rate
        self.device_dict = dict()
        self.boards = dict()
        self.serial_ports = list()
        # self.threads = list()
        self.thread = dict()
        self.thread_all_in_one = None

    def set_serial_ports(self):
        """ Lists serial port names

            :raises EnvironmentError:
                On unsupported or unknown platforms
            :returns:
                A list of the serial ports available on the system
        """
        
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            # this excludes your current terminal "/dev/tty"
            ports = glob.glob('/dev/tty[A-Za-z]*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/tty.*')
        else:
            raise EnvironmentError('Unsupported platform')

        # ipdb.set_trace()
        serial_ports = []
        for port in ports:
            try:
                # command = 'chmod 666 ' + port
                # os.system(command)
                if 'ACM' in port:
                    s = serial.Serial(port)
                    s.close()
                    serial_ports.append(port)
            except (OSError, serial.SerialException):
                pass
        self.serial_ports = serial_ports
    
    def set_serial_ports2(self):
        """ Lists serial port names
            if there are multiple serial comms available other than Arduino
            :returns:
                A list of the serial ports available on the system
        """
        ports = serial.tools.list_ports.comports()

        serial_ports = []
        for port, desc, hwid in ports:
            if('Arduino' in str(desc)):
                serial_ports.append(str(port))
        self.serial_ports = serial_ports

    def device_initialize(self):
        # open the serial ports.
        i=0
        for port in reversed(self.serial_ports):
            self.device_dict[i]={'serial_port': port, 'device_id': i}
            i+=1
        for sys_id in self.device_dict:
            serial_port = self.device_dict[sys_id]['serial_port']
            try:
                self.boards[sys_id] = serial.Serial(serial_port, self.baud_rate, timeout=.1)
                # self.boards[sys_id].isOpen()
                logging.debug("The serial port %s is open.", serial_port)
                time.sleep(0.1)
            except:
                print('Error opening serial port', serial_port)
                traceback.print_exc()

    def send_all(self, packet):
        """Send data to the serial ports.
        @param packet: data to send
        """
        try:
            for sys_id in self.device_dict:
                #while self.anchor_serial_object[sys_id].inWaiting()>0:
                time.sleep(self.tx_wait_time)
                bytes_sent = self.boards[sys_id].write(packet[sys_id])
                logging.debug("%d bytes sent to %s at %f.", bytes_sent, self.device_dict[sys_id]['serial_port'], time.time())
        except:
            print("Sending packets failed:", self.device_dict[sys_id]['serial_port'])
            sys.exit(0)
    
    def send(self, packet, sys_id):
        """Send data to the serial port.
        @param packet: data to send
        @param sys_id: system id of the serial port to send the data to
        """
        try:
            #while self.anchor_serial_object[sys_id].inWaiting()>0:
            time.sleep(self.tx_wait_time)
            bytes_sent = self.boards[sys_id].write(packet[sys_id])
            logging.debug("%d bytes sent to %s at %f.", bytes_sent, self.device_dict[sys_id]['serial_port'], time.time())
        except:
            logging.error("Sending packets failed:", self.device_dict[sys_id]['serial_port'])
            sys.exit(0)

    def create_packet(self, msg):
        """ Generate packet from ris_weights. msg is an weight array vector
        @param msg: the message to send
        @return: byte stream of the message 
        """
        if isinstance(msg, str):
            packet = bytes(msg, 'utf-8')
        elif isinstance(msg, list) and all(isinstance(x, int) for x in msg):
            packet = struct.pack('>%ul' % len(msg), *msg) 
            # print(packet)
        elif isinstance(msg, list):
            packet = bytearray(msg)
        elif isinstance(msg, int):
            packet = struct.pack('>l', msg) 
            # print(packet)
        else:
            logging.error("Message: %s", str(msg)) 
        logging.info(f'Message {msg} created at {time.time()}')  

        return packet
    
    def start_thread(self, packet, sys_id):
        try:
            self.thread[sys_id] = threading.Thread(target=self.send, args=[packet, sys_id])
            self.thread[sys_id].daemon = True
            self.thread[sys_id].start()
            logging.debug(f'Thread {self.thread[sys_id].ident} started at {time.time()}')
        except:
            traceback.print_exc()

    def start_threads_all(self, packet):
        try:
            self.thread_all_in_one = threading.Thread(target=self.send_all, args=[packet])
            self.thread_all_in_one.daemon = True
            self.thread_all_in_one.start()
            logging.debug(f'Thread {self.thread_all_in_one.ident} started at {time.time()}')
        except:
            traceback.print_exc()
    
    def close_connections(self):
        for board in self.boards.values():
            board.close()
            logging.debug("The serial port %s is close.", board.port)
     
    def stop(self):
        #print "Stopping data collectoin."
        if self.thread:
            for t in self.thread.values():
                t.join()
        if self.thread_all_in_one:
            self.thread_all_in_one.join()
        time.sleep(0.2)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# test
if __name__ == '__main__':

    ris = RISController()
    ris.set_serial_ports()
    print(ris.serial_ports)

    # message = {0: ["0","1"],1: ["0","1"]}

    message = {0: 30418120,1:3296720}

    # packet={0: ris.create_packet(message[0][1]), 1: ris.create_packet(message[1][0])}
    packet={0: ris.create_packet(message[0]), 1: ris.create_packet(message[1])}
    
    print(packet)

    print(struct.unpack('>L', packet[0])) 
    print('{0:x}'.format(struct.unpack('>L', packet[0])[0]))

    # ris.device_initialize()
    # print(ris.device_dict)
    # print(ris.boards)


    # # ris.send(packet, 0)
    # for i in range(20):
    #     # num = input("Enter a number: ")
    #     # packet = ris.create_packet(num)
    #     # packet = ris.create_packet(message[0])
    #     # time.sleep(1)
    #     print(f"packet is .{packet}.")
    #     # ris.send(packet)
    #     ris.start_thread(packet, 0)
    #     ris.start_thread(packet, 1)
    #     time.sleep(3)

    #     # data = ris.boards[0].readline()
    #     # print(data)
       

    # target_functions = [(ris.send, (packet, 0))]
    # ris.start_threads(target_functions)
   
    # while True:
    #     ris.send(ris.create_packet(str(message[0])),0)
    #     time.sleep(0.05)
    #     # print(ris.boards[0].readline())
    
    # ris.close_connections()
    
