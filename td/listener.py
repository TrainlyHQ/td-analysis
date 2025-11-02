# Very crude STOMP client from an old Python implementation

import logging
import os
logging.basicConfig(format='%(asctime)s %(levelname)s\t%(message)s', level=logging.INFO)

if __name__ == '__main__':
    logging.info('Starting with PID %s...' % os.getpid())

import stomp
import sys
import time
import socket
import dotenv
import signal
import json
import fastavro

dotenv.load_dotenv("../.env")

env = os.environ

sclass_data_store = {}
update_queue = {}

USERNAME = env.get("NETWORK_RAIL_USERNAME")
PASSWORD = env.get("NETWORK_RAIL_PASSWORD")
HOSTNAME = env.get("NETWORK_RAIL_HOST")
HOSTPORT = env.get("NETWORK_RAIL_PORT")
TOPIC = '/topic/' + env.get("TD_FEED_TOPIC")
AREA_ID=env.get("AREA_ID", "M1")

CCLASS_SCHEMA="../schemas/cclass.json"
SCLASS_SCHEMA="../schemas/sclass.json"

CCLASS_FILE=env.get("CCLASS_FILE", "../data/cclass.avro")
SCLASS_FILE=env.get("SCLASS_FILE", "../data/sclass.avro")

with open(CCLASS_SCHEMA, 'r') as f:
    cclass_schema = json.load(f)
with open(SCLASS_SCHEMA, 'r') as f:
    sclass_schema = json.load(f)

CLIENT_ID = socket.getfqdn()
HEARTBEAT_INTERVAL_MS = 15000
RECONNECT_DELAY_SECS = 15

def exit_handler(signum, frame):
    global shouldExit, shouldStartNewConnection
    logging.info('Exiting...')
    try:
        shouldExit
    except NameError:
        exit(0)
    shouldExit = True
    shouldStartNewConnection = True

def connect_and_subscribe(connection):
    if int(stomp.__version__[0]) < 5:
        connection.start()

    connect_header = {'client-id': USERNAME + '-' + CLIENT_ID}
    subscribe_header = {
        'activemq.subscriptionName': CLIENT_ID + '-TD-LIVE-FEED-DECODER'
    }

    connection.connect(username=USERNAME,
                       passcode=PASSWORD,
                       wait=True,
                       headers=connect_header)

    connection.subscribe(destination=TOPIC,
                         id='1',
                         ack='auto',
                         headers=subscribe_header)


class StompClient(stomp.ConnectionListener):

    def on_heartbeat(self):
        logging.info('Received a heartbeat.')

    def on_heartbeat_timeout(self):
        logging.error('Heartbeat timeout')

    def on_error(self, headers, message):
        logging.error(message)

    def on_disconnected(self):
        logging.warning('Disconnected')
        time.sleep(RECONNECT_DELAY_SECS)

        global shouldExit, shouldStartNewConnection
        shouldExit = True
        shouldStartNewConnection = True

        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    def on_connecting(self, host_and_port):
        logging.info('Connecting to ' + host_and_port[0])

    def on_message(self, frame):
        # If an error occurs, we want to keep processing messages
        try:

            if frame.cmd == 'MESSAGE':
                parsed_json = json.loads(frame.body)
                
                for message in parsed_json:
                    msg_key = list(message.keys())[0]
                    msg_value = message[msg_key]
                    
                    if msg_value['area_id'] != AREA_ID:
                        continue
                    
                    if msg_value['msg_type'][0] == 'C':
                        msg_type = None
                        if msg_value['msg_type'] == 'CA':
                            msg_type = 'step'
                        elif msg_value['msg_type'] == 'CB':
                            msg_type = 'cancel'
                        elif msg_value['msg_type'] == 'CC':
                            msg_type = 'interpose'
                        else:
                            continue

                        cclass_msg = {
                            'timestamp': int(msg_value['time']),
                            'area_id': msg_value['area_id'],
                            'type': msg_type,
                            'from': msg_value['from'] if 'from' in msg_value else None,
                            'to': msg_value['to'] if 'to' in msg_value else None,
                            'train_id': msg_value['descr'],
                        }
                        
                        logging.info(f"C-Class: {cclass_msg['train_id']} Type={cclass_msg['type']} From={cclass_msg['from']} To={cclass_msg['to']}")
                        
                        try:
                            with open(CCLASS_FILE, 'a+b') as fp:
                                fastavro.writer(fp, cclass_schema, [cclass_msg])
                        except FileNotFoundError:
                            with open(CCLASS_FILE, 'wb') as fp:
                                fastavro.writer(fp, cclass_schema, [cclass_msg])
                    elif msg_value['msg_type'][0] == 'S':
                        address = msg_value['address']
                        new_data_hex = msg_value.get('data', '00')
                        
                        # Convert hex to 8-bit binary string
                        new_data_bin = bin(int(new_data_hex, 16))[2:].zfill(8)
                        
                        old_data_bin = sclass_data_store.get(address, None)
                        if old_data_bin is None:
                            sclass_data_store[address] = new_data_bin
                            continue
                        
                        if old_data_bin != new_data_bin:
                            messages_to_write = []
                            for i in range(8):
                                if old_data_bin[i] != new_data_bin[i]:
                                    new_bit_value = int(new_data_bin[i])
                                    # Reverse bit index: 0 is LSB (rightmost), 7 is MSB (leftmost)
                                    bit_index = 7 - i
                                    logging.info(f"S-Class: Address={address}, Bit Index={bit_index}, New Value={new_bit_value}")
                                    
                                    sclass_msg = {
                                        'timestamp': int(msg_value['time']),
                                        'area_id': msg_value['area_id'],
                                        'address': address,
                                        'bit_index': bit_index,
                                        'value': new_bit_value,
                                    }
                                    messages_to_write.append(sclass_msg)
                            
                            if messages_to_write:
                                try:
                                    with open(SCLASS_FILE, 'a+b') as fp:
                                        fastavro.writer(fp, sclass_schema, messages_to_write)
                                except FileNotFoundError:
                                    with open(SCLASS_FILE, 'wb') as fp:
                                        fastavro.writer(fp, sclass_schema, messages_to_write)

                        sclass_data_store[address] = new_data_bin

            else:
                logging.warning('Unknown frame type: %s' % frame.cmd)

        except Exception as e:
            logging.error(str(e))

shouldExit = False
shouldStartNewConnection = False
if __name__ == '__main__':
    signal.signal(signal.SIGINT, exit_handler)

    # Now we can connect
    conn = stomp.Connection12([(HOSTNAME, HOSTPORT)],
                              auto_decode=False,
                              heartbeats=(HEARTBEAT_INTERVAL_MS, HEARTBEAT_INTERVAL_MS),
                              reconnect_sleep_initial=1,
                              reconnect_sleep_increase=2,
                              reconnect_sleep_jitter=0.6,
                              reconnect_sleep_max=60.0,
                              reconnect_attempts_max=60,
                              heart_beat_receive_scale=2.5)
                                  
    conn.set_listener('', StompClient())
    connect_and_subscribe(conn)
    
    while not shouldExit:
        time.sleep(1)

    conn.disconnect()