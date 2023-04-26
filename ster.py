from time import sleep
import airsim as AS
import numpy as np
import tensorflow as tf
from tensorflow import keras
import lib
import cv2 as cv
import torch
print('po imporcie')

IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100

client = AS.CarClient()
client.confirmConnection()
client.enableApiControl(True)
print('airsim połączony')

model = keras.models.load_model('model_sterowanie.h5')
print('ster wczytany')
car_controls = AS.CarControls()
car_controls.throttle = 0.5
client.setCarControls(car_controls)
sleep(3)

yolo = torch.jit.load('data/weights/yolopv2.pt')
yolo = yolo.cuda().half().eval()

print('yolo wczytane')

pierwszy = None

while True:
    img = lib.photo_to_np_ndarray(client)
    img = lib.detect_and_return(yolo, img)
    img = cv.convertScaleAbs(img)
    img = cv.cvtColor(img, cv.COLOR_RGB2HSV)[:,:,1]
    img = cv.resize(img,(IMAGE_HEIGHT,IMAGE_WIDTH))
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.reshape(img, (1, 100, 100, 1))
    direction = model.predict(img)
    if pierwszy is None:
        pierwszy = 'coś'
        continue
    direction = direction[0][0]
    car_controls.steering = direction.astype(float)
    client.setCarControls(car_controls)
    print(direction)