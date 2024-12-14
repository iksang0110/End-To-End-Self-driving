import base64
from datetime import datetime
import os
import shutil
import numpy as np
import subprocess
import utils
import airsim
import keras
from keras.layers import Lambda
from keras.models import load_model
import tensorflow as tf
tf.compat.v1.placeholder


def custom_preprocess(x):
    return x / 127.5 - 1.0


class Road():
    def __init__(self):
        
        self.count = 0
        self.rgb_map = 0.1
        self.prev_frame_time = 0
        self.new_frame_time = 0
        
        self.model_pretrained = load_model('model-010.h5',safe_mode=False)
        
        self.client = airsim.CarClient()
        
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.car_controls = airsim.CarControls()
        
    def getScreenRGB(self):
        responses = self.client.simGetImages([airsim.ImageRequest("2", airsim.ImageType.Scene, False, False)])
        responses1 = self.client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Segmentation, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        if ((responses[0].width != 0 or responses[0].height != 0)):
            rgb = img1d.reshape(response.height, response.width, 3)
           # rgb = cv2.cvtColor(img_rgba, cv2.COLOR_BGR2RGB)
        else:
            print("Something bad happened! Restting AirSim!")
            self.AirSim_reset()
            rgb = np.ones(480, 640, 3)
        

        return rgb
    
    def Drive(self):
        
        image=self.getScreenRGB()
        image = utils.preprocess(image) # apply the preprocessing
        image = np.array([image])       # the model expects 4D array
        steering_angle = float(self.model_pretrained.predict(image, batch_size=1))
        self.car_controls.throttle = 0.3
        self.car_controls.steering = steering_angle
        self.client.setCarControls(self.car_controls)


tensor = Road()
while(1):  
    tensor.Drive()