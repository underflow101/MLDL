##########################################################################
# eval.py
# Dev. Dongwon Paek
#
# Description: Validation test of tflite model file.
##########################################################################

import os, argparse, sys, time
import cv2
import numpy as np
from threading import Thread
import importlib.util

from cv2 import getTickCount, getTickFrequency
import tflite_runtime.interpreter as tflite

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]

def main():
    interpreter = tflite.Interpreter(model_path="converted_model.tflite")
    interpreter.allocate_tensors()
    
    frame_rate_calc = 1
    freq = getTickFrequency()
    
    image_path = []
    
    writing = 0
    phoneWithHand = 0
    others = 0
    sleep = 0

    image_path.append("./others/")
    image_path.append("./phoneWithHand/")
    image_path.append("./sleep/")
    image_path.append("./writing/")
    
    for j in range(0, 4):
        for i in range(0, 100):
            t1 = getTickCount()
            image = cv2.imread(image_path[j] + str(i) + '.jpg')
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Test model on random input data.
            input_shape = input_details[0]['shape']
            input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)

            interpreter.invoke()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            height = input_details[0]['shape'][1]
            width = input_details[0]['shape'][2]

            floating_model = input_details[0]['dtype'] == np.float32
            input_mean = 127.5
            input_std = 127.5
            min_conf_threshold = 0.2
            
            input_data = np.expand_dims(image, axis=0)
            
            output_data = interpreter.get_tensor(output_details[0]['index'])
            results = np.squeeze(output_data)

            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std
            
            top_k = results.argsort()[-4:][::-1]
            labels = ['writing', 'others', 'phoneWithHand', 'sleep']

            tmp = list()
            
            for i in top_k:
                tmp.append('{:08.6f}: {}'.format(float(results[i]), labels[i]))
            if(tmp[0].split(" ")[1] == 'writing'):
                writing += 1
            elif(tmp[0].split(" ")[1] == 'sleep'):
                sleep += 1
            elif(tmp[0].split(" ")[1] == 'phoneWithHand'):
                phoneWithHand += 1
            elif(tmp[0].split(" ")[1] == 'others'):
                others += 1
            
            t2 = getTickCount()
            time1 = (t2 - t1) / freq
            frame_rate_calc = 1 / time1
            print(frame_rate_calc)
                
        print('others: ', end='')
        print(others)
        print('phoneWithHand: ', end='')
        print(phoneWithHand)
        print('writing: ', end='')
        print(writing)
        print('sleep: ', end='')
        print(sleep)
    
if __name__ == '__main__':
    main()