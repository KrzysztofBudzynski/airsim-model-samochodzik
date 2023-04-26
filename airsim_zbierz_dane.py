import airsim
import pandas as pd
from keyboard import is_pressed
from lib import photo_to_np_ndarray
from lib import getVehicleStatus
from PIL import Image
from time import sleep

velocity_list = []
angle_list = []
throttle_list = []
image_names = []
camera_numbers = []
base_name = 'image'
base_path = '../dane/z_airsim/'
i = 0
nr_kamery = 0
record = False
image_dict = {}

client = airsim.CarClient()
client.confirmConnection()
print('zapisywanie na start wyłączone, start/stop pod przyciskiem s')

while True:
    if record:
        img = Image.fromarray(photo_to_np_ndarray(client, nr_kamery))
        image_name = base_name + str(i) + '.jpg'
        image_dict.update({image_name : img})
        i += 1
        (steering, throttle, speed) = getVehicleStatus(client)
        angle_list.append(steering)
        throttle_list.append(throttle)
        image_names.append(image_name)
        camera_numbers.append(nr_kamery)
        velocity_list.append(speed)
    
    if is_pressed('x'):
        sleep(0.1)
        print('gotowe %d obrazów' % i)
    
    if is_pressed('s'):
        sleep(0.5)
        if record:
            record = False
            print('zapisywanie przerwane')
        else:
            record = True
            print('zapisywanie wznowione')
        
    if is_pressed('q'):
        print('zapis...')
        for name, img in image_dict.items():
            img.save(base_path + name)
        data = pd.DataFrame({'nazwa_obrazu' : image_names,
                            'sterowanie' : angle_list,
                            'przyspieszenie' : throttle_list,
                            'predkosc' : velocity_list,
                            'nr_kamery' : camera_numbers})
        data.to_csv(base_path + 'log.csv')
        break