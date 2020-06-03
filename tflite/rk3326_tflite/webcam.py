import os, argparse, sys, time
import cv2
import numpy as np
from threading import Thread
import importlib.util
import tflite_runtime.interpreter as tflite
from cv2 import getTickCount, getTickFrequency

'''
Class: VideoStream
@module:
    - Uses Thread of threading to increase FPS of Raspberry Pi
    - Test has been made from link below:
        - https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
'''
class VideoStream:
    def __init__(self, resolution=(640,480), fps=30):
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
    
    # Uses threading to increase FPS   
    def start(self):
        Thread(target=self.update).start()
        return self

    # If camera is stopped, thread also stops
    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()
    
    # Return frame
    def read(self):
        return self.frame
    
    # Stops camera and threading
    def stop(self):
        self.stopped = True

width, height = (640, 480)

# load tflite model 
interpreter = tflite.Interpreter(model_path="./quant_model_2.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = input_details[0]['dtype'] == np.float32
input_mean = 127.5
input_std = 127.5
min_conf_threshold = 0.2

frame_rate_calc = 1
freq = getTickFrequency()

videostream = VideoStream().start()
time.sleep(1)

while True:
    t1 = getTickCount()
    frame1 = videostream.read()
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    
    if floating_model:
        input_data = (np.float16(input_data) - input_mean) / input_std
        
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    top_k = results.argsort()[-4:][::-1]
    labels = ['writing', 'others', 'phoneWithHand', 'sleep']
    
    '''
    # Class index of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    # Confidence of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    '''
    #cv2.imshow('Classifier', frame)
    for i in top_k:
        print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
    
    t2 = getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1
    print(frame_rate_calc)
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()